// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>
#include <span>

#include "Gpu/GpuVertexAttrib.hpp"
#include "GpuVertexAttribInternal.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class GpuSystemInternalFactory
    {
    public:
        virtual ~GpuSystemInternalFactory();

        virtual std::unique_ptr<GpuVertexAttribsInternal> CreateGpuVertexAttribs(std::span<const GpuVertexAttrib> attribs) const = 0;
    };
} // namespace AIHoloImager
