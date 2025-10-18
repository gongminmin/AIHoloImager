// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandAllocatorInfoInternal.hpp"

namespace AIHoloImager
{
    class D3D12CommandAllocatorInfo : public GpuCommandAllocatorInfoInternal
    {
    public:
        D3D12CommandAllocatorInfo(GpuSystem& gpu_system, GpuSystem::CmdQueueType type);
        ~D3D12CommandAllocatorInfo() noexcept override;

        D3D12CommandAllocatorInfo(D3D12CommandAllocatorInfo&& other) noexcept;
        explicit D3D12CommandAllocatorInfo(GpuCommandAllocatorInfoInternal&& other) noexcept;
        D3D12CommandAllocatorInfo& operator=(D3D12CommandAllocatorInfo&& other) noexcept;
        GpuCommandAllocatorInfoInternal& operator=(GpuCommandAllocatorInfoInternal&& other) noexcept override;

        ID3D12CommandAllocator* NativeCmdAllocator() const noexcept;

        uint64_t FenceValue() const noexcept;
        void FenceValue(uint64_t value) noexcept;

    private:
        ComPtr<ID3D12CommandAllocator> cmd_allocator_;
        uint64_t fence_val_ = 0;
    };
} // namespace AIHoloImager
