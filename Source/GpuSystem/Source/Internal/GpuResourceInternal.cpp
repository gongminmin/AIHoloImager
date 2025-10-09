// Copyright (c) 2025 Minmin Gong
//

#include "GpuResourceInternal.hpp"

namespace AIHoloImager
{
    GpuResourceInternal::GpuResourceInternal() noexcept = default;
    GpuResourceInternal::~GpuResourceInternal() = default;

    GpuResourceInternal::GpuResourceInternal(GpuResourceInternal&& other) noexcept = default;
    GpuResourceInternal& GpuResourceInternal::operator=(GpuResourceInternal&& other) noexcept = default;
} // namespace AIHoloImager
