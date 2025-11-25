// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/InternalDefine.hpp"

namespace AIHoloImager
{
    class GpuCommandPoolInternal;

    class GpuCommandPool
    {
        DISALLOW_COPY_AND_ASSIGN(GpuCommandPool)
        DEFINE_INTERNAL(GpuCommandPool)

    public:
        GpuCommandPool() noexcept;
        GpuCommandPool(GpuSystem& gpu_system, GpuSystem::CmdQueueType type);
        ~GpuCommandPool() noexcept;

        GpuCommandPool(GpuCommandPool&& other) noexcept;
        GpuCommandPool& operator=(GpuCommandPool&& other) noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
