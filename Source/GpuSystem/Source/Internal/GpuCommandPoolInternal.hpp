// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuCommandPoolInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuCommandPoolInternal)

    public:
        GpuCommandPoolInternal() noexcept;
        virtual ~GpuCommandPoolInternal() noexcept;

        GpuCommandPoolInternal(GpuCommandPoolInternal&& other) noexcept;
        virtual GpuCommandPoolInternal& operator=(GpuCommandPoolInternal&& other) noexcept = 0;
    };
} // namespace AIHoloImager
