// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

namespace AIHoloImager
{
    class GpuSystem;
    class GpuCommandAllocatorInfoInternal;

    class GpuCommandAllocatorInfo
    {
    public:
        explicit GpuCommandAllocatorInfo(GpuSystem& gpu_system);
        ~GpuCommandAllocatorInfo() noexcept;

        GpuCommandAllocatorInfoInternal& Internal() noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
