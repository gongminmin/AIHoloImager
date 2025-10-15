// Copyright (c) 2025 Minmin Gong
//

#include "GpuCommandListInternal.hpp"

namespace AIHoloImager
{
    GpuCommandListInternal::GpuCommandListInternal() noexcept = default;
    GpuCommandListInternal::~GpuCommandListInternal() = default;

    GpuCommandListInternal::GpuCommandListInternal(GpuCommandListInternal&& other) noexcept = default;
    GpuCommandListInternal& GpuCommandListInternal::operator=(GpuCommandListInternal&& other) noexcept = default;
} // namespace AIHoloImager
