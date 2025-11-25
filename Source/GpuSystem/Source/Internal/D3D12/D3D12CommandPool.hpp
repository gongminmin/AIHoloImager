// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <list>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Gpu/GpuCommandPool.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandPoolInternal.hpp"
#include "D3D12ImpDefine.hpp"

namespace AIHoloImager
{
    class D3D12CommandPool : public GpuCommandPoolInternal
    {
    public:
        D3D12CommandPool(GpuSystem& gpu_system, GpuSystem::CmdQueueType type);
        ~D3D12CommandPool() noexcept override;

        D3D12CommandPool(D3D12CommandPool&& other) noexcept;
        explicit D3D12CommandPool(GpuCommandPoolInternal&& other) noexcept;
        D3D12CommandPool& operator=(D3D12CommandPool&& other) noexcept;
        GpuCommandPoolInternal& operator=(GpuCommandPoolInternal&& other) noexcept override;

        ID3D12CommandAllocator* CmdAllocator() const noexcept;

        uint64_t FenceValue() const noexcept;
        void FenceValue(uint64_t value) noexcept;

        void RegisterAllocatedCommandList(ID3D12CommandList* cmd_list);
        void UnregisterAllocatedCommandList(ID3D12CommandList* cmd_list);
        bool EmptyAllocatedCommandLists() const noexcept;

    private:
        ComPtr<ID3D12CommandAllocator> cmd_allocator_;
        uint64_t fence_val_ = 0;

        std::list<ID3D12CommandList*> allocated_cmd_lists_;
    };

    D3D12_DEFINE_IMP(CommandPool)
} // namespace AIHoloImager
