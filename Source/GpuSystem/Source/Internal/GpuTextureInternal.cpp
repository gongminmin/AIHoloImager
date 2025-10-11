// Copyright (c) 2025 Minmin Gong
//

#include "GpuTextureInternal.hpp"

namespace AIHoloImager
{
    GpuTextureInternal::GpuTextureInternal() noexcept = default;
    GpuTextureInternal::~GpuTextureInternal() = default;

    GpuTextureInternal::GpuTextureInternal(GpuTextureInternal&& other) noexcept = default;
    GpuTextureInternal& GpuTextureInternal::operator=(GpuTextureInternal&& other) noexcept = default;
} // namespace AIHoloImager
