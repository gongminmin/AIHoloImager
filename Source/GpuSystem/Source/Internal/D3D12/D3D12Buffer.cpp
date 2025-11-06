// Copyright (c) 2025 Minmin Gong
//

#include "D3D12Buffer.hpp"

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "D3D12CommandList.hpp"
#include "D3D12Conversion.hpp"

namespace AIHoloImager
{
    D3D12_RANGE ToD3D12Range(const GpuRange& range)
    {
        return D3D12_RANGE{range.begin, range.end};
    }

    D3D12_IMP_IMP(Buffer)

    D3D12Buffer::D3D12Buffer(GpuSystem& gpu_system, uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name)
        : D3D12Resource(gpu_system), heap_(heap),
          curr_state_(heap == GpuHeap::ReadBack ? GpuResourceState::CopyDst : GpuResourceState::Common)
    {
        this->CreateResource(GpuResourceType::Buffer, size, 1, 1, 1, 1, GpuFormat::Unknown, heap_, flags, curr_state_, std::move(name));
    }
    D3D12Buffer::D3D12Buffer(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name)
        : D3D12Resource(gpu_system, native_resource, std::move(name)), curr_state_(curr_state)
    {
        if (native_resource != nullptr)
        {
            D3D12_HEAP_PROPERTIES heap_prop;
            static_cast<ID3D12Resource*>(native_resource)->GetHeapProperties(&heap_prop, nullptr);
            heap_ = FromD3D12HeapType(heap_prop.Type);
        }
    }

    D3D12Buffer::~D3D12Buffer() = default;

    D3D12Buffer::D3D12Buffer(D3D12Buffer&& other) noexcept = default;
    D3D12Buffer::D3D12Buffer(GpuResourceInternal&& other) noexcept : D3D12Buffer(static_cast<D3D12Buffer&&>(other))
    {
    }
    D3D12Buffer::D3D12Buffer(GpuBufferInternal&& other) noexcept : D3D12Buffer(static_cast<D3D12Buffer&&>(other))
    {
    }
    D3D12Buffer& D3D12Buffer::operator=(D3D12Buffer&& other) noexcept = default;
    GpuResourceInternal& D3D12Buffer::operator=(GpuResourceInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12Buffer&&>(other));
    }
    GpuBufferInternal& D3D12Buffer::operator=(GpuBufferInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12Buffer&&>(other));
    }

    void D3D12Buffer::Name(std::wstring_view name)
    {
        this->D3D12Resource::Name(std::move(name));
    }

    ID3D12Resource* D3D12Buffer::Resource() const noexcept
    {
        return this->D3D12Resource::Resource();
    }

    void* D3D12Buffer::NativeResource() const noexcept
    {
        return this->Resource();
    }

    void* D3D12Buffer::NativeBuffer() const noexcept
    {
        return this->NativeResource();
    }

    void* D3D12Buffer::SharedHandle() const noexcept
    {
        return this->D3D12Resource::SharedHandle();
    }

    GpuResourceType D3D12Buffer::Type() const noexcept
    {
        return this->D3D12Resource::Type();
    }

    uint32_t D3D12Buffer::AllocationSize() const noexcept
    {
        return this->D3D12Resource::AllocationSize();
    }

    GpuVirtualAddressType D3D12Buffer::GpuVirtualAddress() const noexcept
    {
        return this->D3D12Resource::Resource()->GetGPUVirtualAddress();
    }

    uint32_t D3D12Buffer::Size() const noexcept
    {
        return this->Width();
    }

    void* D3D12Buffer::Map(const GpuRange& read_range)
    {
        void* addr;
        const D3D12_RANGE d3d12_read_range = ToD3D12Range(read_range);
        TIFHR(this->D3D12Resource::Resource()->Map(0, &d3d12_read_range, &addr));
        return addr;
    }
    void* D3D12Buffer::Map()
    {
        void* addr;
        const D3D12_RANGE d3d12_read_range{0, 0};
        TIFHR(this->D3D12Resource::Resource()->Map(0, (heap_ == GpuHeap::ReadBack) ? nullptr : &d3d12_read_range, &addr));
        return addr;
    }

    void D3D12Buffer::Unmap(const GpuRange& write_range)
    {
        const D3D12_RANGE d3d12_write_range = ToD3D12Range(write_range);
        this->D3D12Resource::Resource()->Unmap(0, (heap_ == GpuHeap::Upload) ? nullptr : &d3d12_write_range);
    }

    void D3D12Buffer::Unmap()
    {
        this->Unmap(GpuRange{0, 0});
    }

    void D3D12Buffer::Reset()
    {
        D3D12Resource::Reset();
        heap_ = {};
        curr_state_ = {};
    }

    void D3D12Buffer::Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const
    {
        this->Transition(D3D12Imp(cmd_list), sub_resource, target_state);
    }

    void D3D12Buffer::Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const
    {
        this->Transition(D3D12Imp(cmd_list), target_state);
    }

    void D3D12Buffer::Transition(D3D12CommandList& cmd_list, [[maybe_unused]] uint32_t sub_resource, GpuResourceState target_state) const
    {
        assert(sub_resource == 0);
        this->Transition(cmd_list, target_state);
    }

    void D3D12Buffer::Transition(D3D12CommandList& cmd_list, GpuResourceState target_state) const
    {
        auto* native_resource = this->D3D12Resource::Resource();
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
