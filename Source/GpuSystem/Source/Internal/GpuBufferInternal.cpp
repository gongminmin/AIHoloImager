// Copyright (c) 2025 Minmin Gong
//

#include "GpuBufferInternal.hpp"

namespace AIHoloImager
{
    GpuBufferInternal::GpuBufferInternal() noexcept = default;
    GpuBufferInternal::~GpuBufferInternal() = default;

    GpuBufferInternal::GpuBufferInternal(GpuBufferInternal&& other) noexcept = default;
    GpuBufferInternal& GpuBufferInternal::operator=(GpuBufferInternal&& other) noexcept = default;
} // namespace AIHoloImager
