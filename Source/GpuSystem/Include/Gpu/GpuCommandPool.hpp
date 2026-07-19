// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/InternalDefine.hpp"
#include "Gpu/Symbol.hpp"

namespace AIHoloImager
{
    class GpuCommandPoolInternal;

    class GpuCommandPool
    {
        DISALLOW_COPY_AND_ASSIGN(GpuCommandPool)
        DEFINE_INTERNAL(GpuCommandPool)

    public:
        AIHI_GPU_SYS_API GpuCommandPool() noexcept;
        AIHI_GPU_SYS_API GpuCommandPool(GpuSystem& gpu_system, GpuSystem::CmdQueueType type);
        AIHI_GPU_SYS_API ~GpuCommandPool() noexcept;

        AIHI_GPU_SYS_API GpuCommandPool(GpuCommandPool&& other) noexcept;
        AIHI_GPU_SYS_API GpuCommandPool& operator=(GpuCommandPool&& other) noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
