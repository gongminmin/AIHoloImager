// Copyright (c) 2025 Minmin Gong
//

#include "D3D12SystemFactory.hpp"

#include "Internal/D3D12/D3D12VertexAttrib.hpp"

namespace AIHoloImager
{
    D3D12SystemFactory::~D3D12SystemFactory() = default;

    std::unique_ptr<GpuVertexAttribsInternal> D3D12SystemFactory::CreateGpuVertexAttribs(std::span<const GpuVertexAttrib> attribs) const
    {
        return std::make_unique<D3D12VertexAttribs>(std::move(attribs));
    }
} // namespace AIHoloImager
