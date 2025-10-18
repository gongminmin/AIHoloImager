// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    class GpuCommandAllocatorInfoInternal;

    class GpuCommandAllocatorInfo
    {
    public:
        GpuCommandAllocatorInfo(GpuSystem& gpu_system, GpuSystem::CmdQueueType type);
        ~GpuCommandAllocatorInfo() noexcept;

        GpuCommandAllocatorInfoInternal& Internal() noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
