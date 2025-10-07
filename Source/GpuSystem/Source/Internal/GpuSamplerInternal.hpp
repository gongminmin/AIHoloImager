// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuStaticSamplerInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuStaticSamplerInternal)

    public:
        GpuStaticSamplerInternal() noexcept;
        virtual ~GpuStaticSamplerInternal();

        GpuStaticSamplerInternal(GpuStaticSamplerInternal&& other) noexcept;

        virtual GpuStaticSamplerInternal& operator=(GpuStaticSamplerInternal&& other) noexcept = 0;
    };

    class GpuDynamicSamplerInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDynamicSamplerInternal)

    public:
        GpuDynamicSamplerInternal() noexcept;
        virtual ~GpuDynamicSamplerInternal();

        GpuDynamicSamplerInternal(GpuDynamicSamplerInternal&& other) noexcept;

        virtual GpuDynamicSamplerInternal& operator=(GpuDynamicSamplerInternal&& other) noexcept = 0;
    };
} // namespace AIHoloImager
