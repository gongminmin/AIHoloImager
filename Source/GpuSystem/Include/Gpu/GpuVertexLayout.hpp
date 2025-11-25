// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "Gpu/GpuFormat.hpp"
#include "Gpu/InternalDefine.hpp"

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
    class GpuVertexLayoutInternal;

    class GpuVertexLayout
    {
        DEFINE_INTERNAL(GpuVertexLayout)

    public:
        GpuVertexLayout() noexcept;
        GpuVertexLayout(GpuSystem& gpu_system, std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides = {});
        ~GpuVertexLayout() noexcept;

        GpuVertexLayout(const GpuVertexLayout& other);
        GpuVertexLayout& operator=(const GpuVertexLayout& other);

        GpuVertexLayout(GpuVertexLayout&& other) noexcept;
        GpuVertexLayout& operator=(GpuVertexLayout&& other) noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
