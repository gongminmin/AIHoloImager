// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <cstdint>

#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuFenceInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuFenceInternal)

    public:
        GpuFenceInternal() noexcept;
        virtual ~GpuFenceInternal();

        GpuFenceInternal(GpuFenceInternal&& other) noexcept;
        virtual GpuFenceInternal& operator=(GpuFenceInternal&& other) noexcept = 0;

        virtual void* NativeFence() const noexcept = 0;
        virtual void* SharedFenceHandle() const noexcept = 0;

        virtual uint64_t CompletedValue() const = 0;

        virtual void CpuWait(uint64_t value) const = 0;
    };
} // namespace AIHoloImager
