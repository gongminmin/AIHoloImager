// Copyright (c) 2025-2026 Minmin Gong
//

#include "D3D12CommandPool.hpp"

#include "Base/Uuid.hpp"

#include "D3D12Conversion.hpp"
#include "D3D12System.hpp"

DEFINE_UUID_OF(ID3D12CommandAllocator);

namespace AIHoloImager
{
    D3D12_IMP_IMP(CommandPool)

    D3D12CommandPool::D3D12CommandPool(GpuSystem& gpu_system, GpuSystem::CmdQueueType type)
    {
        ID3D12Device* d3d12_device = D3D12Imp(gpu_system).Device();
        TIFHR(
            d3d12_device->CreateCommandAllocator(ToD3D12CommandListType(type), UuidOf<ID3D12CommandAllocator>(), cmd_allocator_.PutVoid()));
    }
    D3D12CommandPool::~D3D12CommandPool() noexcept = default;

    D3D12CommandPool::D3D12CommandPool(D3D12CommandPool&& other) noexcept = default;
    D3D12CommandPool::D3D12CommandPool(GpuCommandPoolInternal&& other) noexcept : D3D12CommandPool(static_cast<D3D12CommandPool&&>(other))
    {
    }
    D3D12CommandPool& D3D12CommandPool::operator=(D3D12CommandPool&& other) noexcept = default;
    GpuCommandPoolInternal& D3D12CommandPool::operator=(GpuCommandPoolInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12CommandPool&&>(other));
    }

    void D3D12CommandPool::Reset()
    {
        TIFHR(cmd_allocator_->Reset());
    }

    ID3D12CommandAllocator* D3D12CommandPool::CmdAllocator() const noexcept
    {
        return cmd_allocator_.Get();
    }

    uint64_t D3D12CommandPool::FenceValue() const noexcept
    {
        return fence_val_;
    }

    void D3D12CommandPool::FenceValue(uint64_t value) noexcept
    {
        fence_val_ = value;
    }

    void D3D12CommandPool::RegisterAllocatedCommandList(ID3D12CommandList* cmd_list)
    {
        allocated_cmd_lists_.push_back(cmd_list);
    }

    void D3D12CommandPool::UnregisterAllocatedCommandList(ID3D12CommandList* cmd_list)
    {
        auto iter = std::find(allocated_cmd_lists_.begin(), allocated_cmd_lists_.end(), cmd_list);
        if (iter != allocated_cmd_lists_.end())
        {
            allocated_cmd_lists_.erase(iter);
        }
    }

    bool D3D12CommandPool::Empty() const noexcept
    {
        return allocated_cmd_lists_.empty();
    }
} // namespace AIHoloImager
