// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Gpu/GpuCommandQueue.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandQueueInternal.hpp"
#include "D3D12ImpDefine.hpp"

namespace AIHoloImager
{
    class D3D12CommandQueue : public GpuCommandQueueInternal
    {
    public:
        D3D12CommandQueue() noexcept;
        D3D12CommandQueue(GpuSystem& gpu_system, GpuSystem::CmdQueueType type, std::string_view name);
        ~D3D12CommandQueue() noexcept override;

        D3D12CommandQueue(D3D12CommandQueue&& other) noexcept;
        explicit D3D12CommandQueue(GpuCommandQueueInternal&& other) noexcept;
        D3D12CommandQueue& operator=(D3D12CommandQueue&& other) noexcept;
        GpuCommandQueueInternal& operator=(GpuCommandQueueInternal&& other) noexcept;

        explicit operator bool() const noexcept override;

        void* NativeCmdQueue() const noexcept override;

        void GpuWait(std::span<const GpuCommandQueue::FenceInfo> fences) override;

        void Execute(const GpuCommandList& cmd_list, std::span<const GpuCommandQueue::FenceInfo> wait_fences,
            const GpuCommandQueue::FenceInfo& signal_fence);
        void Execute(const GpuCommandListInternal& cmd_list_internal, std::span<const GpuCommandQueue::FenceInfo> wait_fences,
            const GpuCommandQueue::FenceInfo& signal_fence) override;

        ID3D12CommandQueue* CmdQueue() const noexcept;

    private:
        ComPtr<ID3D12CommandQueue> cmd_queue_;
    };

    D3D12_DEFINE_IMP(CommandQueue)
} // namespace AIHoloImager
