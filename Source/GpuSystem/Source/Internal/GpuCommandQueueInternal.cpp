// Copyright (c) 2026 Minmin Gong
//

#include "GpuCommandQueueInternal.hpp"

namespace AIHoloImager
{
    GpuCommandQueueInternal::GpuCommandQueueInternal() noexcept = default;
    GpuCommandQueueInternal::~GpuCommandQueueInternal() noexcept = default;

    GpuCommandQueueInternal::GpuCommandQueueInternal(GpuCommandQueueInternal&& other) noexcept = default;
    GpuCommandQueueInternal& GpuCommandQueueInternal::operator=(GpuCommandQueueInternal&& other) noexcept = default;
} // namespace AIHoloImager
