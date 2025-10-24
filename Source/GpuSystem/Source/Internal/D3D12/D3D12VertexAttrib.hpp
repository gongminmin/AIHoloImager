// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <span>
#include <vector>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Gpu/GpuVertexAttrib.hpp"

#include "../GpuVertexAttribInternal.hpp"

namespace AIHoloImager
{
    class D3D12VertexAttribs : public GpuVertexAttribsInternal
    {
    public:
        explicit D3D12VertexAttribs(std::span<const GpuVertexAttrib> attribs);
        ~D3D12VertexAttribs() override;

        D3D12VertexAttribs(const D3D12VertexAttribs& other);
        explicit D3D12VertexAttribs(const GpuVertexAttribsInternal& other);

        D3D12VertexAttribs& operator=(const D3D12VertexAttribs& other);
        GpuVertexAttribsInternal& operator=(const GpuVertexAttribsInternal& other) override;

        D3D12VertexAttribs(D3D12VertexAttribs&& other) noexcept;
        explicit D3D12VertexAttribs(GpuVertexAttribsInternal&& other) noexcept;

        D3D12VertexAttribs& operator=(D3D12VertexAttribs&& other) noexcept;
        GpuVertexAttribsInternal& operator=(GpuVertexAttribsInternal&& other) noexcept override;

        std::unique_ptr<GpuVertexAttribsInternal> Clone() const override;

        std::span<const D3D12_INPUT_ELEMENT_DESC> InputElementDescs() const;

    private:
        void UpdateSemantics();

    private:
        std::vector<D3D12_INPUT_ELEMENT_DESC> input_elems_;
        std::vector<std::string> semantics_;
    };

    const D3D12VertexAttribs& D3D12Imp(const GpuVertexAttribs& vertex_attribs);
} // namespace AIHoloImager
