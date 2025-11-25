// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

namespace AIHoloImager
{
    class GpuVertexLayoutInternal
    {
    public:
        GpuVertexLayoutInternal() noexcept;
        GpuVertexLayoutInternal(const GpuVertexLayoutInternal& other);
        GpuVertexLayoutInternal(GpuVertexLayoutInternal&& other) noexcept;
        virtual ~GpuVertexLayoutInternal();

        virtual GpuVertexLayoutInternal& operator=(const GpuVertexLayoutInternal& other) = 0;
        virtual GpuVertexLayoutInternal& operator=(GpuVertexLayoutInternal&& other) noexcept = 0;

        virtual std::unique_ptr<GpuVertexLayoutInternal> Clone() const = 0;
    };
} // namespace AIHoloImager
