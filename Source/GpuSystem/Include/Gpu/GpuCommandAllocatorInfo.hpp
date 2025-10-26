// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/InternalDefine.hpp"

namespace AIHoloImager
{
    class GpuCommandAllocatorInfoInternal;

    class GpuCommandAllocatorInfo
    {
        DISALLOW_COPY_AND_ASSIGN(GpuCommandAllocatorInfo)
        DEFINE_INTERNAL(GpuCommandAllocatorInfo)

    public:
        GpuCommandAllocatorInfo() noexcept;
        GpuCommandAllocatorInfo(GpuSystem& gpu_system, GpuSystem::CmdQueueType type);
        ~GpuCommandAllocatorInfo() noexcept;

        GpuCommandAllocatorInfo(GpuCommandAllocatorInfo&& other) noexcept;
        GpuCommandAllocatorInfo& operator=(GpuCommandAllocatorInfo&& other) noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
