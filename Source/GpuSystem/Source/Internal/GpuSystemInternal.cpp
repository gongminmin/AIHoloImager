// Copyright (c) 2025 Minmin Gong
//

#include "GpuSystemInternal.hpp"

namespace AIHoloImager
{
    GpuSystemInternal::GpuSystemInternal() noexcept = default;
    GpuSystemInternal::~GpuSystemInternal() = default;

    GpuSystemInternal::GpuSystemInternal(GpuSystemInternal&& other) noexcept = default;
    GpuSystemInternal& GpuSystemInternal::operator=(GpuSystemInternal&& other) noexcept = default;
} // namespace AIHoloImager
