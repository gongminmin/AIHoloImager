// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuCommandAllocatorInfoInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuCommandAllocatorInfoInternal)

    public:
        GpuCommandAllocatorInfoInternal() noexcept;
        virtual ~GpuCommandAllocatorInfoInternal() noexcept;

        GpuCommandAllocatorInfoInternal(GpuCommandAllocatorInfoInternal&& other) noexcept;
        virtual GpuCommandAllocatorInfoInternal& operator=(GpuCommandAllocatorInfoInternal&& other) noexcept = 0;
    };
} // namespace AIHoloImager
