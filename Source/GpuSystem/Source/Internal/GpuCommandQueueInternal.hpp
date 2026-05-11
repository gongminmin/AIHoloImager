// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <span>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuCommandQueue.hpp"
#include "Gpu/GpuFence.hpp"

namespace AIHoloImager
{
    class GpuCommandQueueInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuCommandQueueInternal)

    public:
        GpuCommandQueueInternal() noexcept;
        virtual ~GpuCommandQueueInternal() noexcept;

        GpuCommandQueueInternal(GpuCommandQueueInternal&& other) noexcept;
        virtual GpuCommandQueueInternal& operator=(GpuCommandQueueInternal&& other) noexcept = 0;

        virtual explicit operator bool() const noexcept = 0;

        virtual void* NativeCmdQueue() const noexcept = 0;

        virtual void GpuWait(std::span<const GpuCommandQueue::FenceInfo> fences) = 0;

        virtual void Execute(const GpuCommandList& cmd_list, std::span<const GpuCommandQueue::FenceInfo> wait_fences,
            const GpuCommandQueue::FenceInfo& signal_fence) = 0;
    };
} // namespace AIHoloImager
