// Copyright (c) 2025-2026 Minmin Gong
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

        virtual void Reset() = 0;
        virtual bool Empty() const noexcept = 0;

        virtual uint64_t FenceValue() const noexcept = 0;
        virtual void FenceValue(uint64_t value) noexcept = 0;
    };
} // namespace AIHoloImager
