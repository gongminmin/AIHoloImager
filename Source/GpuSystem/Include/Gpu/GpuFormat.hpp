// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <cstdint>

#include "Gpu/Symbol.hpp"

namespace AIHoloImager
{
    enum class GpuBaseFormat
    {
        Unknown,
        UNorm,
        Uint,
        Sint,
        Float,
    };

    enum class GpuFormat
    {
        Unknown,

        R8_UNorm,
        RG8_UNorm,
        RGBA8_UNorm,
        RGBA8_UNorm_SRGB,
        BGRA8_UNorm,
        BGRA8_UNorm_SRGB,
        BGRX8_UNorm,
        BGRX8_UNorm_SRGB,

        R16_Uint,
        R16_Sint,
        R16_Float,
        RG16_Uint,
        RG16_Sint,
        RG16_Float,
        RGBA16_Uint,
        RGBA16_Sint,
        RGBA16_Float,

        R32_Uint,
        R32_Sint,
        R32_Float,
        RG32_Uint,
        RG32_Sint,
        RG32_Float,
        RGB32_Uint,
        RGB32_Sint,
        RGB32_Float,
        RGBA32_Uint,
        RGBA32_Sint,
        RGBA32_Float,

        D16_UNorm,
        D24_UNorm_S8_Uint,
        D32_Float,
        D32_Float_S8X24_Uint,

        NV12,
    };

    AIHI_GPU_SYS_API uint32_t FormatSize(GpuFormat fmt);
    AIHI_GPU_SYS_API GpuBaseFormat BaseFormat(GpuFormat fmt);
    AIHI_GPU_SYS_API uint32_t FormatChannels(GpuFormat fmt);
    AIHI_GPU_SYS_API uint32_t FormatChannelSize(GpuFormat fmt);
    AIHI_GPU_SYS_API uint32_t NumPlanes(GpuFormat fmt) noexcept;

    AIHI_GPU_SYS_API bool IsDepthStencilFormat(GpuFormat fmt) noexcept;
    AIHI_GPU_SYS_API bool IsStencilFormat(GpuFormat fmt) noexcept;
    AIHI_GPU_SYS_API bool IsSRGBFormat(GpuFormat fmt) noexcept;

    AIHI_GPU_SYS_API GpuFormat ToLinearFormat(GpuFormat fmt) noexcept;
    AIHI_GPU_SYS_API GpuFormat ToSRGBFormat(GpuFormat fmt) noexcept;
} // namespace AIHoloImager
