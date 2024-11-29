// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <directx/d3d12.h>

#include "GpuFormat.hpp"

namespace AIHoloImager
{
    struct GpuVertexAttrib
    {
        static constexpr uint32_t AppendOffset = ~0U;

        std::string semantic;
        uint32_t semantic_index;
        GpuFormat format;
        uint32_t slot = 0;
        uint32_t offset = AppendOffset;
    };

    class GpuVertexAttribs
    {
    public:
        explicit GpuVertexAttribs(std::span<const GpuVertexAttrib> attribs);

        GpuVertexAttribs(const GpuVertexAttribs& other) noexcept;
        GpuVertexAttribs& operator=(const GpuVertexAttribs& other) noexcept;

        GpuVertexAttribs(GpuVertexAttribs&& other) noexcept;
        GpuVertexAttribs& operator=(GpuVertexAttribs&& other) noexcept;

        std::span<const D3D12_INPUT_ELEMENT_DESC> InputElementDescs() const;

    private:
        void UpdateSemantics();

    private:
        std::vector<D3D12_INPUT_ELEMENT_DESC> input_elems_;
        std::vector<std::string> semantics_;
    };
} // namespace AIHoloImager
