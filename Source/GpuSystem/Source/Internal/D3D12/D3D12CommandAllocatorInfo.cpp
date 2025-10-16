// Copyright (c) 2025 Minmin Gong
//

#include "D3D12CommandAllocatorInfo.hpp"

namespace AIHoloImager
{
    D3D12CommandAllocatorInfo::D3D12CommandAllocatorInfo() noexcept = default;
    D3D12CommandAllocatorInfo::~D3D12CommandAllocatorInfo() noexcept = default;

    D3D12CommandAllocatorInfo::D3D12CommandAllocatorInfo(D3D12CommandAllocatorInfo&& other) noexcept = default;
    D3D12CommandAllocatorInfo::D3D12CommandAllocatorInfo(GpuCommandAllocatorInfoInternal&& other) noexcept
        : D3D12CommandAllocatorInfo(std::forward<D3D12CommandAllocatorInfo>(static_cast<D3D12CommandAllocatorInfo&&>(other)))
    {
    }
    D3D12CommandAllocatorInfo& D3D12CommandAllocatorInfo::operator=(D3D12CommandAllocatorInfo&& other) noexcept = default;
    GpuCommandAllocatorInfoInternal& D3D12CommandAllocatorInfo::operator=(GpuCommandAllocatorInfoInternal&& other) noexcept
    {
        return this->operator=(std::move(static_cast<D3D12CommandAllocatorInfo&&>(other)));
    }

    ComPtr<ID3D12CommandAllocator>& D3D12CommandAllocatorInfo::CmdAllocator() noexcept
    {
        return cmd_allocator_;
    }

    uint64_t D3D12CommandAllocatorInfo::FenceValue() const noexcept
    {
        return fence_val_;
    }

    void D3D12CommandAllocatorInfo::FenceValue(uint64_t value) noexcept
    {
        fence_val_ = value;
    }
} // namespace AIHoloImager
