// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include <memory>

#include "Gpu/GpuVertexLayout.hpp"

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

        virtual std::span<const GpuVertexAttrib> Attribs() const noexcept = 0;
        virtual std::span<const uint32_t> SlotStrides() const noexcept = 0;
    };
} // namespace AIHoloImager
