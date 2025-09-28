// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuTexture.hpp"

#include <span>

#include <directx/d3d12.h>

#include "Base/Util.hpp"
#include "Gpu/D3D12/D3D12Traits.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSystem.hpp"

#include "D3D12/D3D12Conversion.hpp"

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

    uint32_t CalcSubResource(
        uint32_t mip_slice, uint32_t array_slice, uint32_t plane_slice, uint32_t num_mip_levels, uint32_t array_size) noexcept
    {
        return (plane_slice * array_size + array_slice) * num_mip_levels + mip_slice;
    }


    GpuTexture::GpuTexture() = default;

    GpuTexture::GpuTexture(GpuSystem& gpu_system, GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
        uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
        : GpuResource(gpu_system), format_(format), flags_(flags)
    {
        if (mip_levels == 0)
        {
            mip_levels = LogNextPowerOf2(std::max({width, height, depth}));
        }

        curr_states_.resize(array_size * mip_levels * NumPlanes(format), GpuResourceState::Common);

        uint32_t depth_or_array_size;
        if (type == GpuResourceType::Texture3D)
        {
            assert(array_size == 1);
            depth_or_array_size = depth;
        }
        else
        {
            assert(type != GpuResourceType::Buffer);
            assert(depth == 1);
            depth_or_array_size = array_size;
        }

        this->CreateResource(
            type, width, height, depth, array_size, mip_levels, format, GpuHeap::Default, flags_, curr_states_[0], std::move(name));
    }

    GpuTexture::GpuTexture(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name) noexcept
        : GpuResource(gpu_system, native_resource, std::move(name))
    {
        if (this->NativeResource() != nullptr)
        {
            curr_states_.assign(this->MipLevels() * this->Planes(), curr_state);
            format_ = GpuResource::Format();
            flags_ = GpuResource::Flags();
        }
    }

    GpuTexture::~GpuTexture() = default;

    GpuTexture::GpuTexture(GpuTexture&& other) noexcept = default;
    GpuTexture& GpuTexture::operator=(GpuTexture&& other) noexcept = default;

    void* GpuTexture::NativeTexture() const noexcept
    {
        return this->NativeResource();
    }

    uint32_t GpuTexture::Width(uint32_t mip) const noexcept
    {
        return std::max(GpuResource::Width() >> mip, 1U);
    }

    uint32_t GpuTexture::Height(uint32_t mip) const noexcept
    {
        return std::max(GpuResource::Height() >> mip, 1U);
    }

    uint32_t GpuTexture::Depth(uint32_t mip) const noexcept
    {
        return std::max(GpuResource::Depth() >> mip, 1U);
    }

    uint32_t GpuTexture::ArraySize() const noexcept
    {
        return GpuResource::ArraySize();
    }

    uint32_t GpuTexture::MipLevels() const noexcept
    {
        return GpuResource::MipLevels();
    }

    uint32_t GpuTexture::Planes() const noexcept
    {
        return NumPlanes(format_);
    }

    GpuFormat GpuTexture::Format() const noexcept
    {
        return format_;
    }

    GpuResourceFlag GpuTexture::Flags() const noexcept
    {
        return flags_;
    }

    void GpuTexture::Reset()
    {
        GpuResource::Reset();
        curr_states_.clear();
    }

    void GpuTexture::Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const
    {
        const D3D12_RESOURCE_STATES d3d12_target_state = ToD3D12ResourceState(target_state);
        if (curr_states_[sub_resource] != target_state)
        {
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = this->NativeResource<D3D12Traits>();
            barrier.Transition.StateBefore = ToD3D12ResourceState(curr_states_[sub_resource]);
            barrier.Transition.StateAfter = d3d12_target_state;
            barrier.Transition.Subresource = sub_resource;
            cmd_list.Transition(std::span(&barrier, 1));

            curr_states_[sub_resource] = target_state;
        }
    }

    void GpuTexture::Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const
    {
        auto* native_resource = this->NativeResource<D3D12Traits>();
        const D3D12_RESOURCE_STATES d3d12_target_state = ToD3D12ResourceState(target_state);
        if ((curr_states_[0] == target_state) &&
            ((target_state == GpuResourceState::UnorderedAccess) || (target_state == GpuResourceState::RayTracingAS)))
        {
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.UAV.pResource = native_resource;
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
                    barrier.Transition.pResource = native_resource;
                    barrier.Transition.StateBefore = ToD3D12ResourceState(curr_states_[0]);
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
                    if (curr_states_[i] != target_state)
                    {
                        auto& barrier = barriers.emplace_back();
                        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                        barrier.Transition.pResource = native_resource;
                        barrier.Transition.StateBefore = ToD3D12ResourceState(curr_states_[i]);
                        barrier.Transition.StateAfter = d3d12_target_state;
                        barrier.Transition.Subresource = static_cast<uint32_t>(i);
                    }
                }
                cmd_list.Transition(std::span(barriers.begin(), barriers.end()));
            }
        }

        curr_states_.assign(this->MipLevels() * this->Planes(), target_state);
    }


    GpuTexture2D::GpuTexture2D() = default;

    GpuTexture2D::GpuTexture2D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t mip_levels, GpuFormat format,
        GpuResourceFlag flags, std::wstring_view name)
        : GpuTexture(gpu_system, GpuResourceType::Texture2D, width, height, 1, 1, mip_levels, format, flags, std::move(name))
    {
    }

    GpuTexture2D::GpuTexture2D(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name) noexcept
        : GpuTexture(gpu_system, native_resource, curr_state, std::move(name))
    {
    }

    GpuTexture2D::GpuTexture2D(GpuTexture2D&& other) noexcept = default;
    GpuTexture2D& GpuTexture2D::operator=(GpuTexture2D&& other) noexcept = default;


    GpuTexture2DArray::GpuTexture2DArray() = default;

    GpuTexture2DArray::GpuTexture2DArray(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t array_size, uint32_t mip_levels,
        GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
        : GpuTexture(gpu_system, GpuResourceType::Texture2DArray, width, height, 1, array_size, mip_levels, format, flags, std::move(name))
    {
    }

    GpuTexture2DArray::GpuTexture2DArray(
        GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name) noexcept
        : GpuTexture(gpu_system, native_resource, curr_state, std::move(name))
    {
    }

    GpuTexture2DArray::GpuTexture2DArray(GpuTexture2DArray&& other) noexcept = default;
    GpuTexture2DArray& GpuTexture2DArray::operator=(GpuTexture2DArray&& other) noexcept = default;


    GpuTexture3D::GpuTexture3D() = default;

    GpuTexture3D::GpuTexture3D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t depth, uint32_t mip_levels,
        GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
        : GpuTexture(gpu_system, GpuResourceType::Texture3D, width, height, depth, 1, mip_levels, format, flags, std::move(name))
    {
    }

    GpuTexture3D::GpuTexture3D(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name) noexcept
        : GpuTexture(gpu_system, native_resource, curr_state, std::move(name))
    {
    }

    GpuTexture3D::GpuTexture3D(GpuTexture3D&& other) noexcept = default;
    GpuTexture3D& GpuTexture3D::operator=(GpuTexture3D&& other) noexcept = default;
} // namespace AIHoloImager
