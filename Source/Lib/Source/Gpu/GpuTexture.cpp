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
            return 2;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8X8_UNORM:
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        case DXGI_FORMAT_R16G16_SINT:
            return 4;

        default:
            // TODO: Support more formats
            AI_HOLO_IMAGER_UNREACHABLE("Unsupported format");
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

    void SubResourceToMipLevelPlane(uint32_t sub_resource, uint32_t num_mip_levels, uint32_t& mip, uint32_t& plane)
    {
        plane = sub_resource / num_mip_levels;
        mip = sub_resource - plane * num_mip_levels;
    }


    GpuTexture2D::GpuTexture2D() = default;

    GpuTexture2D::GpuTexture2D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t mip_levels, DXGI_FORMAT format,
        D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES init_state, std::wstring_view name)
        : curr_states_(mip_levels * NumPlanes(format), init_state)
    {
        const D3D12_HEAP_PROPERTIES default_heap_prop = {
            D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};

        desc_ = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, static_cast<uint64_t>(width), height, 1, static_cast<uint16_t>(mip_levels), format,
            {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, flags};
        TIFHR(gpu_system.NativeDevice()->CreateCommittedResource(
            &default_heap_prop, D3D12_HEAP_FLAG_NONE, &desc_, init_state, nullptr, UuidOf<ID3D12Resource>(), resource_.PutVoid()));
        if (!name.empty())
        {
            resource_->SetName(std::wstring(name).c_str());
        }
    }

    GpuTexture2D::GpuTexture2D(ID3D12Resource* native_resource, D3D12_RESOURCE_STATES curr_state, std::wstring_view name) noexcept
        : resource_(native_resource, false)
    {
        if (resource_)
        {
            desc_ = resource_->GetDesc();
            if (!name.empty())
            {
                resource_->SetName(std::wstring(name).c_str());
            }

            curr_states_.assign(this->MipLevels() * this->Planes(), curr_state);
        }
    }

    GpuTexture2D::~GpuTexture2D() noexcept = default;
    GpuTexture2D::GpuTexture2D(GpuTexture2D&& other) noexcept = default;
    GpuTexture2D& GpuTexture2D::operator=(GpuTexture2D&& other) noexcept = default;

    GpuTexture2D GpuTexture2D::Share() const
    {
        GpuTexture2D texture;
        texture.resource_ = resource_;
        texture.desc_ = desc_;
        texture.curr_states_ = curr_states_;
        return texture;
    }

    GpuTexture2D::operator bool() const noexcept
    {
        return resource_ ? true : false;
    }

    ID3D12Resource* GpuTexture2D::NativeTexture() const noexcept
    {
        return resource_.Get();
    }

    uint32_t GpuTexture2D::Width(uint32_t mip) const noexcept
    {
        return std::max(static_cast<uint32_t>(desc_.Width >> mip), 1U);
    }

    uint32_t GpuTexture2D::Height(uint32_t mip) const noexcept
    {
        return std::max(desc_.Height >> mip, 1U);
    }

    uint32_t GpuTexture2D::MipLevels() const noexcept
    {
        return desc_.MipLevels;
    }

    uint32_t GpuTexture2D::Planes() const noexcept
    {
        return NumPlanes(desc_.Format);
    }

    DXGI_FORMAT GpuTexture2D::Format() const noexcept
    {
        return desc_.Format;
    }

    D3D12_RESOURCE_FLAGS GpuTexture2D::Flags() const noexcept
    {
        return desc_.Flags;
    }

    void GpuTexture2D::Reset() noexcept
    {
        resource_ = nullptr;
        desc_ = {};
        curr_states_.clear();
    }

    D3D12_RESOURCE_STATES GpuTexture2D::State(uint32_t sub_resource) const noexcept
    {
        return curr_states_[sub_resource];
    }

    void GpuTexture2D::Transition(GpuCommandList& cmd_list, uint32_t sub_resource, D3D12_RESOURCE_STATES target_state) const
    {
        if (curr_states_[sub_resource] != target_state)
        {
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = resource_.Get();
            barrier.Transition.StateBefore = curr_states_[sub_resource];
            barrier.Transition.StateAfter = target_state;
            barrier.Transition.Subresource = sub_resource;
            cmd_list.Transition(std::span(&barrier, 1));

            curr_states_[sub_resource] = target_state;
        }
    }

    void GpuTexture2D::Transition(GpuCommandList& cmd_list, D3D12_RESOURCE_STATES target_state) const
    {
        if ((curr_states_[0] == target_state) && ((target_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
                                                     (target_state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)))
        {
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.UAV.pResource = resource_.Get();
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
                if (curr_states_[0] != target_state)
                {
                    D3D12_RESOURCE_BARRIER barrier;
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                    barrier.Transition.pResource = resource_.Get();
                    barrier.Transition.StateBefore = curr_states_[0];
                    barrier.Transition.StateAfter = target_state;
                    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    cmd_list.Transition(std::span(&barrier, 1));
                }
            }
            else
            {
                std::vector<D3D12_RESOURCE_BARRIER> barriers;
                for (size_t i = 0; i < curr_states_.size(); ++i)
                {
                    if (curr_states_[i] != target_state)
                    {
                        auto& barrier = barriers.emplace_back();
                        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                        barrier.Transition.pResource = resource_.Get();
                        barrier.Transition.StateBefore = curr_states_[i];
                        barrier.Transition.StateAfter = target_state;
                        barrier.Transition.Subresource = static_cast<uint32_t>(i);
                    }
                }
                cmd_list.Transition(std::span(barriers.begin(), barriers.end()));
            }
        }

        curr_states_.assign(this->MipLevels() * this->Planes(), target_state);
    }

    void GpuTexture2D::Upload(GpuSystem& gpu_system, GpuCommandList& cmd_list, uint32_t sub_resource, const void* data)
    {
        const uint32_t mip = sub_resource % this->MipLevels();
        const uint32_t width = this->Width(mip);
        const uint32_t height = this->Height(mip);
        const uint32_t format_size = FormatSize(this->Format());

        auto* d3d12_device = gpu_system.NativeDevice();

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
        uint32_t num_row = 0;
        uint64_t row_size_in_bytes = 0;
        uint64_t required_size = 0;
        d3d12_device->GetCopyableFootprints(&desc_, sub_resource, 1, 0, &layout, &num_row, &row_size_in_bytes, &required_size);

        auto upload_mem_block =
            gpu_system.AllocUploadMemBlock(static_cast<uint32_t>(required_size), GpuMemoryAllocator::TextureDataAligment);

        assert(row_size_in_bytes >= width * format_size);

        uint8_t* tex_data = upload_mem_block.CpuAddress<uint8_t>();
        for (uint32_t y = 0; y < height; ++y)
        {
            memcpy(tex_data + y * row_size_in_bytes, reinterpret_cast<const uint8_t*>(data) + y * width * format_size, width * format_size);
        }

        layout.Offset += upload_mem_block.Offset();
        D3D12_TEXTURE_COPY_LOCATION src;
        src.pResource = upload_mem_block.NativeBuffer();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = layout;

        D3D12_TEXTURE_COPY_LOCATION dst;
        dst.pResource = resource_.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = sub_resource;

        D3D12_BOX src_box;
        src_box.left = 0;
        src_box.top = 0;
        src_box.front = 0;
        src_box.right = width;
        src_box.bottom = height;
        src_box.back = 1;

        assert((cmd_list.Type() == GpuSystem::CmdQueueType::Render) || (cmd_list.Type() == GpuSystem::CmdQueueType::Compute));
        auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

        auto src_old_state = this->State(0);
        this->Transition(cmd_list, D3D12_RESOURCE_STATE_COPY_DEST);

        d3d12_cmd_list->CopyTextureRegion(&dst, 0, 0, 0, &src, &src_box);

        this->Transition(cmd_list, src_old_state);

        gpu_system.DeallocUploadMemBlock(std::move(upload_mem_block));
    }

    void GpuTexture2D::Readback(GpuSystem& gpu_system, GpuCommandList& cmd_list, uint32_t sub_resource, void* data) const
    {
        const uint32_t mip = sub_resource % this->MipLevels();
        const uint32_t width = this->Width(mip);
        const uint32_t height = this->Height(mip);
        const uint32_t format_size = FormatSize(this->Format());

        auto* d3d12_device = gpu_system.NativeDevice();

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
        uint32_t num_row = 0;
        uint64_t row_size_in_bytes = 0;
        uint64_t required_size = 0;
        d3d12_device->GetCopyableFootprints(&desc_, sub_resource, 1, 0, &layout, &num_row, &row_size_in_bytes, &required_size);

        auto readback_mem_block =
            gpu_system.AllocReadbackMemBlock(static_cast<uint32_t>(required_size), GpuMemoryAllocator::TextureDataAligment);

        D3D12_TEXTURE_COPY_LOCATION src;
        src.pResource = resource_.Get();
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
        src_box.back = 1;

        assert((cmd_list.Type() == GpuSystem::CmdQueueType::Render) || (cmd_list.Type() == GpuSystem::CmdQueueType::Compute));
        auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

        auto src_old_state = this->State(0);
        this->Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);

        d3d12_cmd_list->CopyTextureRegion(&dst, 0, 0, 0, &src, &src_box);

        this->Transition(cmd_list, src_old_state);

        gpu_system.ExecuteAndReset(cmd_list);
        gpu_system.WaitForGpu();

        assert(row_size_in_bytes >= width * format_size);

        uint8_t* u8_data = reinterpret_cast<uint8_t*>(data);
        const uint8_t* tex_data = readback_mem_block.CpuAddress<uint8_t>();
        for (uint32_t y = 0; y < height; ++y)
        {
            memcpy(&u8_data[y * width * format_size], tex_data + y * row_size_in_bytes, width * format_size);
        }

        gpu_system.DeallocReadbackMemBlock(std::move(readback_mem_block));
    }

    void GpuTexture2D::CopyFrom(GpuSystem& gpu_system, GpuCommandList& cmd_list, const GpuTexture2D& other, uint32_t sub_resource,
        uint32_t dst_x, uint32_t dst_y, const D3D12_BOX& src_box)
    {
        D3D12_TEXTURE_COPY_LOCATION src;
        src.pResource = other.resource_.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = sub_resource;

        D3D12_TEXTURE_COPY_LOCATION dst;
        dst.pResource = resource_.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = sub_resource;

        assert((cmd_list.Type() == GpuSystem::CmdQueueType::Render) || (cmd_list.Type() == GpuSystem::CmdQueueType::Compute));
        auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

        auto src_old_state = other.State(0);
        auto dst_old_state = this->State(0);
        other.Transition(cmd_list, D3D12_RESOURCE_STATE_COPY_SOURCE);
        this->Transition(cmd_list, D3D12_RESOURCE_STATE_COPY_DEST);

        d3d12_cmd_list->CopyTextureRegion(&dst, dst_x, dst_y, 0, &src, &src_box);

        other.Transition(cmd_list, src_old_state);
        this->Transition(cmd_list, dst_old_state);

        gpu_system.ExecuteAndReset(cmd_list);
    }
} // namespace AIHoloImager
