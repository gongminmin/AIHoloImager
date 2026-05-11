// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuFence.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/InternalDefine.hpp"

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
        GpuCommandQueue() noexcept;
        GpuCommandQueue(GpuSystem& gpu_system, GpuSystem::CmdQueueType type, std::string_view name);
        ~GpuCommandQueue() noexcept;

        GpuCommandQueue(GpuCommandQueue&& other) noexcept;
        GpuCommandQueue& operator=(GpuCommandQueue&& other) noexcept;

        explicit operator bool() const noexcept;

        void* NativeCmdQueue() const noexcept;

        void GpuWait(std::span<const FenceInfo> fences);

        void Execute(const GpuCommandList& cmd_list, std::span<const FenceInfo> wait_fences, const FenceInfo& signal_fence);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
