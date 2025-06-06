// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuTexture.hpp"

#include <span>

#include <directx/d3d12.h>

#include "Base/ErrorHandling.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
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
        uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
        : GpuResource(gpu_system), curr_states_(array_size * mip_levels * NumPlanes(format), D3D12_RESOURCE_STATE_COMMON), format_(format)
    {
        ID3D12Device* d3d12_device = gpu_system.NativeDevice();

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
            static_cast<uint16_t>(mip_levels), ToDxgiFormat(format), {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, ToD3D12ResourceFlags(flags)};
        TIFHR(d3d12_device->CreateCommittedResource(&default_heap_prop, ToD3D12HeapFlags(flags), &desc_, curr_states_[0], nullptr,
            UuidOf<ID3D12Resource>(), resource_.Object().PutVoid()));
        this->Name(std::move(name));

        this->CreateSharedHandle(gpu_system, flags);
    }

    GpuTexture::GpuTexture(
        GpuSystem& gpu_system, ID3D12Resource* native_resource, GpuResourceState curr_state, std::wstring_view name) noexcept
        : GpuResource(gpu_system, native_resource)
    {
        if (resource_)
        {
            desc_ = resource_->GetDesc();
            this->Name(std::move(name));

            curr_states_.assign(this->MipLevels() * this->Planes(), ToD3D12ResourceState(curr_state));
        }
    }

    GpuTexture::~GpuTexture() = default;

    GpuTexture::GpuTexture(GpuTexture&& other) noexcept = default;
    GpuTexture& GpuTexture::operator=(GpuTexture&& other) noexcept = default;

    ID3D12Resource* GpuTexture::NativeTexture() const noexcept
    {
        return this->NativeResource();
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
        return NumPlanes(format_);
    }

    GpuFormat GpuTexture::Format() const noexcept
    {
        return format_;
    }

    D3D12_RESOURCE_FLAGS GpuTexture::Flags() const noexcept
    {
        return desc_.Flags;
    }

    void GpuTexture::Reset()
    {
        GpuResource::Reset();
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


    GpuTexture2D::GpuTexture2D() = default;

    GpuTexture2D::GpuTexture2D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t mip_levels, GpuFormat format,
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
        GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
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
        GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
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
