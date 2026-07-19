// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuFence.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/InternalDefine.hpp"
#include "Gpu/Symbol.hpp"

namespace AIHoloImager
{
    class GpuCommandQueueInternal;

    class GpuCommandQueue
    {
        DISALLOW_COPY_AND_ASSIGN(GpuCommandQueue)
        DEFINE_INTERNAL(GpuCommandQueue)

    public:
        struct FenceInfo
        {
            const GpuFence* fence;
            uint64_t value;
        };

    public:
        AIHI_GPU_SYS_API GpuCommandQueue() noexcept;
        AIHI_GPU_SYS_API GpuCommandQueue(GpuSystem& gpu_system, GpuSystem::CmdQueueType type, std::string_view name);
        AIHI_GPU_SYS_API ~GpuCommandQueue() noexcept;

        AIHI_GPU_SYS_API GpuCommandQueue(GpuCommandQueue&& other) noexcept;
        AIHI_GPU_SYS_API GpuCommandQueue& operator=(GpuCommandQueue&& other) noexcept;

        AIHI_GPU_SYS_API explicit operator bool() const noexcept;

        AIHI_GPU_SYS_API void* NativeCmdQueue() const noexcept;

        AIHI_GPU_SYS_API void GpuWait(std::span<const FenceInfo> fences);

        AIHI_GPU_SYS_API void Execute(
            const GpuCommandList& cmd_list, std::span<const FenceInfo> wait_fences, const FenceInfo& signal_fence);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
