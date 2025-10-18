// Copyright (c) 2025 Minmin Gong
//

#include "GpuVertexAttribInternal.hpp"

namespace AIHoloImager
{
    GpuVertexAttribsInternal::GpuVertexAttribsInternal() noexcept = default;
    GpuVertexAttribsInternal::~GpuVertexAttribsInternal() = default;

    GpuVertexAttribsInternal::GpuVertexAttribsInternal(const GpuVertexAttribsInternal& other) = default;
    GpuVertexAttribsInternal& GpuVertexAttribsInternal::operator=(const GpuVertexAttribsInternal& other) = default;

    GpuVertexAttribsInternal::GpuVertexAttribsInternal(GpuVertexAttribsInternal&& other) noexcept = default;
    GpuVertexAttribsInternal& GpuVertexAttribsInternal::operator=(GpuVertexAttribsInternal&& other) noexcept = default;
} // namespace AIHoloImager
