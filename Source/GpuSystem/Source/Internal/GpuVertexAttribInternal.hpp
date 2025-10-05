// Copyright (c) 2025 Minmin Gong
//

#pragma once

namespace AIHoloImager
{
    class GpuVertexAttribsInternal
    {
    public:
        virtual ~GpuVertexAttribsInternal();

        virtual GpuVertexAttribsInternal& operator=(const GpuVertexAttribsInternal& other) = 0;
        virtual GpuVertexAttribsInternal& operator=(GpuVertexAttribsInternal&& other) noexcept = 0;

        virtual std::unique_ptr<GpuVertexAttribsInternal> Clone() const = 0;
    };
} // namespace AIHoloImager
