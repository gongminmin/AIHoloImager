// Copyright (c) 2026 Minmin Gong
//

#include "GpuFenceInternal.hpp"

namespace AIHoloImager
{
    GpuFenceInternal::GpuFenceInternal() noexcept = default;
    GpuFenceInternal::~GpuFenceInternal() = default;

    GpuFenceInternal::GpuFenceInternal(GpuFenceInternal&& other) noexcept = default;
    GpuFenceInternal& GpuFenceInternal::operator=(GpuFenceInternal&& other) noexcept = default;
} // namespace AIHoloImager
