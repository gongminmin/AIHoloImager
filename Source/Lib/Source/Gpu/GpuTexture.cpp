// Copyright (c) 2024 Minmin Gong
//

#include "GpuTexture.hpp"

#include <span>

#include <directx/d3d12.h>

#include "GpuCommandList.hpp"
#include "GpuSystem.hpp"
#include "Util/ErrorHandling.hpp"

namespace AIHoloImager
{
    uint32_t FormatSize(DXGI_FORMAT fmt) noexcept
    {
        switch (fmt)
        {
        case DXGI_FORMAT_R8_UNORM:
            return 1;

        case DXGI_FORMAT_R8G8_UNORM:
        case DXGI_FORMAT_R16_UINT:
            return 2;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8X8_UNORM:
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        case DXGI_FORMAT_R16G16_SINT:
        case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_R32_SINT:
        case DXGI_FORMAT_R32_FLOAT:
            return 4;

        case DXGI_FORMAT_R32G32_UINT:
        case DXGI_FORMAT_R32G32_SINT:
        case DXGI_FORMAT_R32G32_FLOAT:
            return 8;

        case DXGI_FORMAT_R32G32B32A32_UINT:
        case DXGI_FORMAT_R32G32B32A32_SINT:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return 16;

        default:
            // TODO: Support more formats
            Unreachable("Unsupported format");
        }
    }

    uint32_t NumPlanes(DXGI_FORMAT fmt) noexcept
    {
        switch (fmt)
        {
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        case DXGI_FORMAT_NV12:
            // TODO: Support more formats
            return 2;

        default:
            return 1;
        }
    }

    void DecomposeSubResource(uint32_t sub_resource, uint32_t num_mip_levels, uint32_t array_size, uint32_t& mip_slice,
        uint32_t& array_slice, uint32_t& plane_slice) noexcept
    {
        const uint32_t plane_array = sub_resource / num_mip_levels;
        mip_slice = sub_resource - plane_array * num_mip_levels;
        plane_slice = plane_array / array_size;
        array_slice = plane_array - plane_slice * array_size;
    }

    uint32_t CalcSubresource(
        uint32_t mip_slice, uint32_t array_slice, uint32_t plane_slice, uint32_t num_mip_levels, uint32_t array_size) noexcept
    {
        return (plane_slice * array_size + array_slice) * num_mip_levels + mip_slice;
    }


    GpuTexture::GpuTexture() = default;

    GpuTexture::GpuTexture(GpuSystem& gpu_system, D3D12_RESOURCE_DIMENSION dim, uint32_t width, uint32_t height, uint32_t depth,
        uint32_t array_size, uint32_t mip_levels, DXGI_FORMAT format, GpuResourceFlag flags, std::wstring_view name)
        : resource_(gpu_system, nullptr), curr_states_(array_size * mip_levels * NumPlanes(format), D3D12_RESOURCE_STATE_COMMON)
    {
        const D3D12_HEAP_PROPERTIES default_heap_prop = {
            D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};

        uint32_t depth_or_array_size;
        if (dim == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        {
            assert(depth == 1);
            depth_or_array_size = array_size;
        }
        else
        {
            assert(dim == D3D12_RESOURCE_DIMENSION_TEXTURE3D);
            assert(array_size == 1);
            depth_or_array_size = depth;
        }

        desc_ = {dim, 0, static_cast<uint64_t>(width), height, static_cast<uint16_t>(depth_or_array_size),
            static_cast<uint16_t>(mip_levels), format, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, ToD3D12ResourceFlags(flags)};
        TIFHR(gpu_system.NativeDevice()->CreateCommittedResource(&default_heap_prop, D3D12_HEAP_FLAG_NONE, &desc_, curr_states_[0], nullptr,
            UuidOf<ID3D12Resource>(), resource_.Object().PutVoid()));
        if (!name.empty())
        {
            resource_->SetName(std::wstring(name).c_str());
        }
    }

    GpuTexture::GpuTexture(
        GpuSystem& gpu_system, ID3D12Resource* native_resource, GpuResourceState curr_state, std::wstring_view name) noexcept
        : resource_(gpu_system, ComPtr<ID3D12Resource>(native_resource, false))
    {
        if (resource_)
        {
            desc_ = resource_->GetDesc();
            if (!name.empty())
            {
                resource_->SetName(std::wstring(name).c_str());
            }

            curr_states_.assign(this->MipLevels() * this->Planes(), ToD3D12ResourceState(curr_state));
        }
    }

    GpuTexture::~GpuTexture() = default;

    GpuTexture::GpuTexture(GpuTexture&& other) noexcept = default;
    GpuTexture& GpuTexture::operator=(GpuTexture&& other) noexcept = default;

    GpuTexture::operator bool() const noexcept
    {
        return resource_ ? true : false;
    }

    ID3D12Resource* GpuTexture::NativeTexture() const noexcept
    {
        return resource_.Object().Get();
    }

    uint32_t GpuTexture::Width(uint32_t mip) const noexcept
    {
        return std::max(static_cast<uint32_t>(desc_.Width >> mip), 1U);
    }

    uint32_t GpuTexture::Height(uint32_t mip) const noexcept
    {
        return std::max(desc_.Height >> mip, 1U);
    }

    uint32_t GpuTexture::Depth(uint32_t mip) const noexcept
    {
        if (desc_.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        {
            return 1;
        }
        else
        {
            assert(desc_.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D);
            return std::max(desc_.DepthOrArraySize >> mip, 1);
        }
    }

    uint32_t GpuTexture::ArraySize() const noexcept
    {
        if (desc_.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        {
            return desc_.DepthOrArraySize;
        }
        else
        {
            assert(desc_.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D);
            return 1;
        }
    }

    uint32_t GpuTexture::MipLevels() const noexcept
    {
        return desc_.MipLevels;
    }

    uint32_t GpuTexture::Planes() const noexcept
    {
        return NumPlanes(desc_.Format);
    }

    DXGI_FORMAT GpuTexture::Format() const noexcept
    {
        return desc_.Format;
    }

    D3D12_RESOURCE_FLAGS GpuTexture::Flags() const noexcept
    {
        return desc_.Flags;
    }

    void GpuTexture::Reset()
    {
        resource_.Reset();
        desc_ = {};
        curr_states_.clear();
    }

    void GpuTexture::Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const
    {
        const D3D12_RESOURCE_STATES d3d12_target_state = ToD3D12ResourceState(target_state);
        if (curr_states_[sub_resource] != d3d12_target_state)
        {
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = resource_.Object().Get();
            barrier.Transition.StateBefore = curr_states_[sub_resource];
            barrier.Transition.StateAfter = d3d12_target_state;
            barrier.Transition.Subresource = sub_resource;
            cmd_list.Transition(std::span(&barrier, 1));

            curr_states_[sub_resource] = d3d12_target_state;
        }
    }

    void GpuTexture::Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const
    {
        const D3D12_RESOURCE_STATES d3d12_target_state = ToD3D12ResourceState(target_state);
        if ((curr_states_[0] == d3d12_target_state) &&
            ((target_state == GpuResourceState::UnorderedAccess) || (target_state == GpuResourceState::RayTracingAS)))
        {
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.UAV.pResource = resource_.Object().Get();
            cmd_list.Transition(std::span(&barrier, 1));
        }
        else
        {
            bool same_state = true;
            for (size_t i = 1; i < curr_states_.size(); ++i)
            {
                if (curr_states_[i] != curr_states_[0])
                {
                    same_state = false;
                    break;
                }
            }

            if (same_state)
            {
                if (curr_states_[0] != d3d12_target_state)
                {
                    D3D12_RESOURCE_BARRIER barrier;
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                    barrier.Transition.pResource = resource_.Object().Get();
                    barrier.Transition.StateBefore = curr_states_[0];
                    barrier.Transition.StateAfter = d3d12_target_state;
                    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    cmd_list.Transition(std::span(&barrier, 1));
                }
            }
            else
            {
                std::vector<D3D12_RESOURCE_BARRIER> barriers;
                for (size_t i = 0; i < curr_states_.size(); ++i)
                {
                    if (curr_states_[i] != d3d12_target_state)
                    {
                        auto& barrier = barriers.emplace_back();
                        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                        barrier.Transition.pResource = resource_.Object().Get();
                        barrier.Transition.StateBefore = curr_states_[i];
                        barrier.Transition.StateAfter = d3d12_target_state;
                        barrier.Transition.Subresource = static_cast<uint32_t>(i);
                    }
                }
                cmd_list.Transition(std::span(barriers.begin(), barriers.end()));
            }
        }

        curr_states_.assign(this->MipLevels() * this->Planes(), d3d12_target_state);
    }

    void GpuTexture::Upload(GpuSystem& gpu_system, GpuCommandList& cmd_list, uint32_t sub_resource, const void* data)
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, this->MipLevels(), this->ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = this->Width(mip);
        const uint32_t height = this->Height(mip);
        const uint32_t depth = this->Depth(mip);
        const uint32_t format_size = FormatSize(this->Format());

        auto* d3d12_device = gpu_system.NativeDevice();

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
        uint64_t required_size = 0;
        d3d12_device->GetCopyableFootprints(&desc_, sub_resource, 1, 0, &layout, nullptr, nullptr, &required_size);

        auto upload_mem_block =
            gpu_system.AllocUploadMemBlock(static_cast<uint32_t>(required_size), GpuMemoryAllocator::TextureDataAligment);

        assert(layout.Footprint.RowPitch >= width * format_size);

        uint8_t* tex_data = upload_mem_block.CpuAddress<uint8_t>();
        for (uint32_t z = 0; z < depth; ++z)
        {
            for (uint32_t y = 0; y < height; ++y)
            {
                std::memcpy(tex_data + (z * layout.Footprint.Height + y) * layout.Footprint.RowPitch,
                    reinterpret_cast<const uint8_t*>(data) + (z * height + y) * width * format_size, width * format_size);
            }
        }

        layout.Offset += upload_mem_block.Offset();
        D3D12_TEXTURE_COPY_LOCATION src;
        src.pResource = upload_mem_block.NativeBuffer();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = layout;

        D3D12_TEXTURE_COPY_LOCATION dst;
        dst.pResource = resource_.Object().Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = sub_resource;

        D3D12_BOX src_box;
        src_box.left = 0;
        src_box.top = 0;
        src_box.front = 0;
        src_box.right = width;
        src_box.bottom = height;
        src_box.back = depth;

        assert((cmd_list.Type() == GpuSystem::CmdQueueType::Render) || (cmd_list.Type() == GpuSystem::CmdQueueType::Compute));
        auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

        this->Transition(cmd_list, GpuResourceState::CopyDst);

        d3d12_cmd_list->CopyTextureRegion(&dst, 0, 0, 0, &src, &src_box);

        gpu_system.DeallocUploadMemBlock(std::move(upload_mem_block));
    }

    void GpuTexture::Readback(GpuSystem& gpu_system, GpuCommandList& cmd_list, uint32_t sub_resource, void* data) const
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, this->MipLevels(), this->ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = this->Width(mip);
        const uint32_t height = this->Height(mip);
        const uint32_t depth = this->Depth(mip);
        const uint32_t format_size = FormatSize(this->Format());

        auto* d3d12_device = gpu_system.NativeDevice();

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
        uint64_t required_size = 0;
        d3d12_device->GetCopyableFootprints(&desc_, sub_resource, 1, 0, &layout, nullptr, nullptr, &required_size);

        auto readback_mem_block =
            gpu_system.AllocReadbackMemBlock(static_cast<uint32_t>(required_size), GpuMemoryAllocator::TextureDataAligment);

        D3D12_TEXTURE_COPY_LOCATION src;
        src.pResource = resource_.Object().Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = sub_resource;

        layout.Offset = readback_mem_block.Offset();
        D3D12_TEXTURE_COPY_LOCATION dst;
        dst.pResource = readback_mem_block.NativeBuffer();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = layout;

        D3D12_BOX src_box;
        src_box.left = 0;
        src_box.top = 0;
        src_box.front = 0;
        src_box.right = width;
        src_box.bottom = height;
        src_box.back = depth;

        assert((cmd_list.Type() == GpuSystem::CmdQueueType::Render) || (cmd_list.Type() == GpuSystem::CmdQueueType::Compute));
        auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

        this->Transition(cmd_list, GpuResourceState::CopySrc);

        d3d12_cmd_list->CopyTextureRegion(&dst, 0, 0, 0, &src, &src_box);

        gpu_system.ExecuteAndReset(cmd_list);
        gpu_system.WaitForGpu();

        assert(layout.Footprint.RowPitch >= width * format_size);

        uint8_t* u8_data = reinterpret_cast<uint8_t*>(data);
        const uint8_t* tex_data = readback_mem_block.CpuAddress<uint8_t>();
        for (uint32_t z = 0; z < depth; ++z)
        {
            for (uint32_t y = 0; y < height; ++y)
            {
                std::memcpy(&u8_data[(z * height + y) * width * format_size],
                    tex_data + (z * layout.Footprint.Height + y) * layout.Footprint.RowPitch, width * format_size);
            }
        }

        gpu_system.DeallocReadbackMemBlock(std::move(readback_mem_block));
    }

    void GpuTexture::CopyFrom(GpuSystem& gpu_system, GpuCommandList& cmd_list, const GpuTexture& other, uint32_t sub_resource,
        uint32_t dst_x, uint32_t dst_y, uint32_t dst_z, const D3D12_BOX& src_box)
    {
        D3D12_TEXTURE_COPY_LOCATION src;
        src.pResource = other.resource_.Object().Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = sub_resource;

        D3D12_TEXTURE_COPY_LOCATION dst;
        dst.pResource = resource_.Object().Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = sub_resource;

        assert((cmd_list.Type() == GpuSystem::CmdQueueType::Render) || (cmd_list.Type() == GpuSystem::CmdQueueType::Compute));
        auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

        other.Transition(cmd_list, GpuResourceState::CopySrc);
        this->Transition(cmd_list, GpuResourceState::CopyDst);

        d3d12_cmd_list->CopyTextureRegion(&dst, dst_x, dst_y, dst_z, &src, &src_box);

        gpu_system.ExecuteAndReset(cmd_list);
    }


    GpuTexture2D::GpuTexture2D() = default;

    GpuTexture2D::GpuTexture2D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t mip_levels, DXGI_FORMAT format,
        GpuResourceFlag flags, std::wstring_view name)
        : GpuTexture(gpu_system, D3D12_RESOURCE_DIMENSION_TEXTURE2D, width, height, 1, 1, mip_levels, format, flags, std::move(name))
    {
    }

    GpuTexture2D::GpuTexture2D(GpuTexture2D&& other) noexcept = default;
    GpuTexture2D& GpuTexture2D::operator=(GpuTexture2D&& other) noexcept = default;

    GpuTexture2D GpuTexture2D::Share() const
    {
        GpuTexture2D texture;
        texture.resource_ = resource_.Share();
        texture.desc_ = desc_;
        texture.curr_states_ = curr_states_;
        return texture;
    }


    GpuTexture2DArray::GpuTexture2DArray() = default;

    GpuTexture2DArray::GpuTexture2DArray(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t array_size, uint32_t mip_levels,
        DXGI_FORMAT format, GpuResourceFlag flags, std::wstring_view name)
        : GpuTexture(
              gpu_system, D3D12_RESOURCE_DIMENSION_TEXTURE2D, width, height, 1, array_size, mip_levels, format, flags, std::move(name))
    {
    }

    GpuTexture2DArray::GpuTexture2DArray(GpuTexture2DArray&& other) noexcept = default;
    GpuTexture2DArray& GpuTexture2DArray::operator=(GpuTexture2DArray&& other) noexcept = default;

    GpuTexture2DArray GpuTexture2DArray::Share() const
    {
        GpuTexture2DArray texture;
        texture.resource_ = resource_.Share();
        texture.desc_ = desc_;
        texture.curr_states_ = curr_states_;
        return texture;
    }


    GpuTexture3D::GpuTexture3D() = default;

    GpuTexture3D::GpuTexture3D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t depth, uint32_t mip_levels,
        DXGI_FORMAT format, GpuResourceFlag flags, std::wstring_view name)
        : GpuTexture(gpu_system, D3D12_RESOURCE_DIMENSION_TEXTURE3D, width, height, depth, 1, mip_levels, format, flags, std::move(name))
    {
    }

    GpuTexture3D::GpuTexture3D(GpuTexture3D&& other) noexcept = default;
    GpuTexture3D& GpuTexture3D::operator=(GpuTexture3D&& other) noexcept = default;

    GpuTexture3D GpuTexture3D::Share() const
    {
        GpuTexture3D texture;
        texture.resource_ = resource_.Share();
        texture.desc_ = desc_;
        texture.curr_states_ = curr_states_;
        return texture;
    }
} // namespace AIHoloImager
