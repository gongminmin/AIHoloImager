// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuBuffer.hpp"

#include "Base/ErrorHandling.hpp"
#include "Gpu/D3D12/D3D12Traits.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"

#include "D3D12/D3D12Conversion.hpp"

namespace AIHoloImager
{
    D3D12_RANGE ToD3D12Range(const GpuRange& range)
    {
        return D3D12_RANGE{range.begin, range.end};
    }


    GpuBuffer::GpuBuffer() noexcept = default;

    GpuBuffer::GpuBuffer(GpuSystem& gpu_system, uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name)
        : GpuResource(gpu_system), heap_(heap),
          curr_state_(heap == GpuHeap::ReadBack ? GpuResourceState::CopyDst : GpuResourceState::Common)
    {
        this->CreateResource(GpuResourceType::Buffer, size, 1, 1, 1, 1, GpuFormat::Unknown, heap_, flags, curr_state_, std::move(name));
    }

    GpuBuffer::GpuBuffer(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name)
        : GpuResource(gpu_system, native_resource, std::move(name)), curr_state_(curr_state)
    {
        auto* resource = this->NativeBuffer<D3D12Traits>();
        if (resource != nullptr)
        {
            D3D12_HEAP_PROPERTIES heap_prop;
            resource->GetHeapProperties(&heap_prop, nullptr);
            heap_ = FromD3D12HeapType(heap_prop.Type);
        }
    }

    GpuBuffer::~GpuBuffer() = default;

    GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept = default;
    GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept = default;

    void* GpuBuffer::NativeBuffer() const noexcept
    {
        return this->NativeResource();
    }

    GpuVirtualAddressType GpuBuffer::GpuVirtualAddress() const noexcept
    {
        return this->NativeBuffer<D3D12Traits>()->GetGPUVirtualAddress();
    }

    uint32_t GpuBuffer::Size() const noexcept
    {
        return this->Width();
    }

    void* GpuBuffer::Map(const GpuRange& read_range)
    {
        void* addr;
        const D3D12_RANGE d3d12_read_range = ToD3D12Range(read_range);
        TIFHR(this->NativeBuffer<D3D12Traits>()->Map(0, &d3d12_read_range, &addr));
        return addr;
    }

    const void* GpuBuffer::Map(const GpuRange& read_range) const
    {
        return const_cast<GpuBuffer*>(this)->Map(read_range);
    }

    void* GpuBuffer::Map()
    {
        void* addr;
        const D3D12_RANGE d3d12_read_range{0, 0};
        TIFHR(this->NativeBuffer<D3D12Traits>()->Map(0, (heap_ == GpuHeap::ReadBack) ? nullptr : &d3d12_read_range, &addr));
        return addr;
    }

    const void* GpuBuffer::Map() const
    {
        return const_cast<GpuBuffer*>(this)->Map();
    }

    void GpuBuffer::Unmap(const GpuRange& write_range)
    {
        const D3D12_RANGE d3d12_write_range = ToD3D12Range(write_range);
        this->NativeBuffer<D3D12Traits>()->Unmap(0, (heap_ == GpuHeap::Upload) ? nullptr : &d3d12_write_range);
    }

    void GpuBuffer::Unmap()
    {
        this->Unmap(GpuRange{0, 0});
    }

    void GpuBuffer::Unmap() const
    {
        return const_cast<GpuBuffer*>(this)->Unmap();
    }

    void GpuBuffer::Reset()
    {
        GpuResource::Reset();
        heap_ = {};
        curr_state_ = {};
    }

    void GpuBuffer::Transition(GpuCommandList& cmd_list, [[maybe_unused]] uint32_t sub_resource, GpuResourceState target_state) const
    {
        assert(sub_resource == 0);
        this->Transition(cmd_list, target_state);
    }

    void GpuBuffer::Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const
    {
        auto* native_resource = this->NativeResource<D3D12Traits>();
        const D3D12_RESOURCE_STATES d3d12_target_state = ToD3D12ResourceState(target_state);

        D3D12_RESOURCE_BARRIER barrier;
        if (curr_state_ != target_state)
        {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = native_resource;
            barrier.Transition.StateBefore = ToD3D12ResourceState(curr_state_);
            barrier.Transition.StateAfter = d3d12_target_state;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmd_list.Transition(std::span(&barrier, 1));
        }
        else if ((target_state == GpuResourceState::UnorderedAccess) || (target_state == GpuResourceState::RayTracingAS))
        {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.UAV.pResource = native_resource;
            cmd_list.Transition(std::span(&barrier, 1));
        }

        curr_state_ = target_state;
    }
} // namespace AIHoloImager
