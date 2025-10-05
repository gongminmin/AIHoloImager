// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>
#include <span>

#include "../GpuSystemInternalFactory.hpp"
#include "Gpu/GpuVertexAttrib.hpp"

namespace AIHoloImager
{
    class D3D12SystemFactory : public GpuSystemInternalFactory
    {
    public:
        ~D3D12SystemFactory() override;

        std::unique_ptr<GpuVertexAttribsInternal> CreateGpuVertexAttribs(std::span<const GpuVertexAttrib> attribs) const override;
    };
} // namespace AIHoloImager
