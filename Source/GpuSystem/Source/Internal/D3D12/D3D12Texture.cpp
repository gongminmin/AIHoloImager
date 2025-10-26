// Copyright (c) 2025 Minmin Gong
//

#include "D3D12Texture.hpp"

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/Util.hpp"

#include "D3D12/D3D12Conversion.hpp"
#include "D3D12CommandList.hpp"

namespace AIHoloImager
{
    D3D12_IMP_IMP(Texture)

    D3D12Texture::D3D12Texture(GpuSystem& gpu_system, GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
        uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
        : D3D12Resource(gpu_system), format_(format), flags_(flags)
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

    D3D12Texture::D3D12Texture(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name)
        : D3D12Resource(gpu_system, native_resource, std::move(name))
    {
        if (this->NativeResource() != nullptr)
        {
            curr_states_.assign(this->MipLevels() * this->Planes(), curr_state);
            format_ = this->D3D12Resource::Format();
            flags_ = this->D3D12Resource::Flags();
        }
    }

    D3D12Texture::~D3D12Texture() = default;

    D3D12Texture::D3D12Texture(D3D12Texture&& other) noexcept = default;
    D3D12Texture::D3D12Texture(GpuResourceInternal&& other) noexcept : D3D12Texture(static_cast<D3D12Texture&&>(other))
    {
    }
    D3D12Texture::D3D12Texture(GpuTextureInternal&& other) noexcept : D3D12Texture(static_cast<D3D12Texture&&>(other))
    {
    }
    D3D12Texture& D3D12Texture::operator=(D3D12Texture&& other) noexcept = default;
    GpuResourceInternal& D3D12Texture::operator=(GpuResourceInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12Texture&&>(other));
    }
    GpuTextureInternal& D3D12Texture::operator=(GpuTextureInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12Texture&&>(other));
    }

    void D3D12Texture::Name(std::wstring_view name)
    {
        this->D3D12Resource::Name(name);
    }

    void* D3D12Texture::NativeResource() const noexcept
    {
        return this->D3D12Resource::Resource();
    }

    void* D3D12Texture::NativeTexture() const noexcept
    {
        return this->NativeResource();
    }

    void* D3D12Texture::SharedHandle() const noexcept
    {
        return this->D3D12Resource::SharedHandle();
    }

    GpuResourceType D3D12Texture::Type() const noexcept
    {
        return this->D3D12Resource::Type();
    }

    uint32_t D3D12Texture::Width(uint32_t mip) const noexcept
    {
        return std::max(this->D3D12Resource::Width() >> mip, 1U);
    }

    uint32_t D3D12Texture::Height(uint32_t mip) const noexcept
    {
        return std::max(this->D3D12Resource::Height() >> mip, 1U);
    }

    uint32_t D3D12Texture::Depth(uint32_t mip) const noexcept
    {
        return std::max(this->D3D12Resource::Depth() >> mip, 1U);
    }

    uint32_t D3D12Texture::ArraySize() const noexcept
    {
        return this->D3D12Resource::ArraySize();
    }

    uint32_t D3D12Texture::MipLevels() const noexcept
    {
        return this->D3D12Resource::MipLevels();
    }

    uint32_t D3D12Texture::Planes() const noexcept
    {
        return NumPlanes(format_);
    }

    GpuFormat D3D12Texture::Format() const noexcept
    {
        return format_;
    }

    GpuResourceFlag D3D12Texture::Flags() const noexcept
    {
        return flags_;
    }

    void D3D12Texture::Reset()
    {
        this->D3D12Resource::Reset();
        curr_states_.clear();
    }

    void D3D12Texture::Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const
    {
        this->Transition(D3D12Imp(cmd_list), sub_resource, target_state);
    }

    void D3D12Texture::Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const
    {
        this->Transition(D3D12Imp(cmd_list), target_state);
    }

    void D3D12Texture::Transition(D3D12CommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const
    {
        const D3D12_RESOURCE_STATES d3d12_target_state = ToD3D12ResourceState(target_state);
        if (curr_states_[sub_resource] != target_state)
        {
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = this->D3D12Resource::Resource();
            barrier.Transition.StateBefore = ToD3D12ResourceState(curr_states_[sub_resource]);
            barrier.Transition.StateAfter = d3d12_target_state;
            barrier.Transition.Subresource = sub_resource;
            cmd_list.Transition(std::span(&barrier, 1));

            curr_states_[sub_resource] = target_state;
        }
    }

    void D3D12Texture::Transition(D3D12CommandList& cmd_list, GpuResourceState target_state) const
    {
        auto* native_resource = this->D3D12Resource::Resource();
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
} // namespace AIHoloImager
