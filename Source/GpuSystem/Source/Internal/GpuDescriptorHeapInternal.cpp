// Copyright (c) 2025 Minmin Gong
//

#include "GpuDescriptorHeapInternal.hpp"

namespace AIHoloImager
{
    GpuDescriptorHeapInternal::GpuDescriptorHeapInternal() noexcept = default;
    GpuDescriptorHeapInternal::~GpuDescriptorHeapInternal() = default;

    GpuDescriptorHeapInternal::GpuDescriptorHeapInternal(GpuDescriptorHeapInternal&& other) noexcept = default;
    GpuDescriptorHeapInternal& GpuDescriptorHeapInternal::operator=(GpuDescriptorHeapInternal&& other) noexcept = default;
} // namespace AIHoloImager
