// Copyright (c) 2025 Minmin Gong
//

#include "GpuCommandAllocatorInfoInternal.hpp"

namespace AIHoloImager
{
    GpuCommandAllocatorInfoInternal::GpuCommandAllocatorInfoInternal() noexcept = default;
    GpuCommandAllocatorInfoInternal::~GpuCommandAllocatorInfoInternal() noexcept = default;

    GpuCommandAllocatorInfoInternal::GpuCommandAllocatorInfoInternal(GpuCommandAllocatorInfoInternal&& other) noexcept = default;
    GpuCommandAllocatorInfoInternal& GpuCommandAllocatorInfoInternal::operator=(GpuCommandAllocatorInfoInternal&& other) noexcept = default;
} // namespace AIHoloImager
