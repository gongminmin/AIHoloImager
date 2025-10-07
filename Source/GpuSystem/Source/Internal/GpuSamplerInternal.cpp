// Copyright (c) 2025 Minmin Gong
//

#include "GpuSamplerInternal.hpp"

namespace AIHoloImager
{
    GpuStaticSamplerInternal::GpuStaticSamplerInternal() noexcept = default;
    GpuStaticSamplerInternal::~GpuStaticSamplerInternal() = default;

    GpuStaticSamplerInternal::GpuStaticSamplerInternal(GpuStaticSamplerInternal&& other) noexcept = default;
    GpuStaticSamplerInternal& GpuStaticSamplerInternal::operator=(GpuStaticSamplerInternal&& other) noexcept = default;


    GpuDynamicSamplerInternal::GpuDynamicSamplerInternal() noexcept = default;
    GpuDynamicSamplerInternal::~GpuDynamicSamplerInternal() = default;

    GpuDynamicSamplerInternal::GpuDynamicSamplerInternal(GpuDynamicSamplerInternal&& other) noexcept = default;
    GpuDynamicSamplerInternal& GpuDynamicSamplerInternal::operator=(GpuDynamicSamplerInternal&& other) noexcept = default;
} // namespace AIHoloImager
