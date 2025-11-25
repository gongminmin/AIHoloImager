// Copyright (c) 2025 Minmin Gong
//

#include "GpuCommandPoolInternal.hpp"

namespace AIHoloImager
{
    GpuCommandPoolInternal::GpuCommandPoolInternal() noexcept = default;
    GpuCommandPoolInternal::~GpuCommandPoolInternal() noexcept = default;

    GpuCommandPoolInternal::GpuCommandPoolInternal(GpuCommandPoolInternal&& other) noexcept = default;
    GpuCommandPoolInternal& GpuCommandPoolInternal::operator=(GpuCommandPoolInternal&& other) noexcept = default;
} // namespace AIHoloImager
