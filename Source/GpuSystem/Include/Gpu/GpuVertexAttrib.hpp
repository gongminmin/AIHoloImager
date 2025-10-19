// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "Gpu/GpuFormat.hpp"

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

    class GpuSystem;
    class GpuVertexAttribsInternal;

    class GpuVertexAttribs
    {
    public:
        GpuVertexAttribs() noexcept;
        GpuVertexAttribs(GpuSystem& gpu_system, std::span<const GpuVertexAttrib> attribs);
        ~GpuVertexAttribs() noexcept;

        GpuVertexAttribs(const GpuVertexAttribs& other);
        GpuVertexAttribs& operator=(const GpuVertexAttribs& other);

        GpuVertexAttribs(GpuVertexAttribs&& other) noexcept;
        GpuVertexAttribs& operator=(GpuVertexAttribs&& other) noexcept;

        const GpuVertexAttribsInternal& Internal() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
