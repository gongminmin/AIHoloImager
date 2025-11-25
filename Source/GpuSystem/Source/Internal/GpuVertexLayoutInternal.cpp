// Copyright (c) 2025 Minmin Gong
//

#include "GpuVertexLayoutInternal.hpp"

namespace AIHoloImager
{
    GpuVertexLayoutInternal::GpuVertexLayoutInternal() noexcept = default;
    GpuVertexLayoutInternal::~GpuVertexLayoutInternal() = default;

    GpuVertexLayoutInternal::GpuVertexLayoutInternal(const GpuVertexLayoutInternal& other) = default;
    GpuVertexLayoutInternal& GpuVertexLayoutInternal::operator=(const GpuVertexLayoutInternal& other) = default;

    GpuVertexLayoutInternal::GpuVertexLayoutInternal(GpuVertexLayoutInternal&& other) noexcept = default;
    GpuVertexLayoutInternal& GpuVertexLayoutInternal::operator=(GpuVertexLayoutInternal&& other) noexcept = default;
} // namespace AIHoloImager
