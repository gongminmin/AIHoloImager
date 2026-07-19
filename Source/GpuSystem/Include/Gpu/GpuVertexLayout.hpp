// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "Gpu/GpuFormat.hpp"
#include "Gpu/InternalDefine.hpp"
#include "Gpu/Symbol.hpp"

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
        static const uint32_t InvalidIndex = ~0U;

    public:
        AIHI_GPU_SYS_API GpuVertexLayout() noexcept;
        AIHI_GPU_SYS_API GpuVertexLayout(
            GpuSystem& gpu_system, std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides = {});
        AIHI_GPU_SYS_API ~GpuVertexLayout() noexcept;

        AIHI_GPU_SYS_API GpuVertexLayout(const GpuVertexLayout& other);
        AIHI_GPU_SYS_API GpuVertexLayout& operator=(const GpuVertexLayout& other);

        AIHI_GPU_SYS_API GpuVertexLayout(GpuVertexLayout&& other) noexcept;
        AIHI_GPU_SYS_API GpuVertexLayout& operator=(GpuVertexLayout&& other) noexcept;

        AIHI_GPU_SYS_API std::span<const GpuVertexAttrib> Attribs() const noexcept;
        AIHI_GPU_SYS_API std::span<const uint32_t> SlotStrides() const noexcept;

        AIHI_GPU_SYS_API uint32_t FindAttrib(std::string_view semantic, uint32_t index) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
