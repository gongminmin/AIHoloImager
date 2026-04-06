// Copyright (c) 2026 Minmin Gong
//

#include "GpuQueryInternal.hpp"

namespace AIHoloImager
{
    GpuTimerQueryInternal::GpuTimerQueryInternal() noexcept = default;
    GpuTimerQueryInternal::~GpuTimerQueryInternal() = default;

    GpuTimerQueryInternal::GpuTimerQueryInternal(GpuTimerQueryInternal&& other) noexcept = default;
    GpuTimerQueryInternal& GpuTimerQueryInternal::operator=(GpuTimerQueryInternal&& other) noexcept = default;
} // namespace AIHoloImager
