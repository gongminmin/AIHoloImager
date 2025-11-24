// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuFormat.hpp"

#include "Base/ErrorHandling.hpp"

namespace AIHoloImager
{
    uint32_t FormatSize(GpuFormat fmt)
    {
        switch (fmt)
        {
        case GpuFormat::D24_UNorm_S8_Uint:
            return 4;
        case GpuFormat::D32_Float_S8X24_Uint:
            return 8;

        default:
            return FormatChannelSize(fmt) * FormatChannels(fmt);
        }
    }

    GpuBaseFormat BaseFormat(GpuFormat fmt)
    {
        switch (fmt)
        {
        case GpuFormat::Unknown:
            return GpuBaseFormat::Unknown;

        case GpuFormat::R8_UNorm:
        case GpuFormat::RG8_UNorm:
        case GpuFormat::RGBA8_UNorm:
        case GpuFormat::RGBA8_UNorm_SRGB:
        case GpuFormat::BGRA8_UNorm:
        case GpuFormat::BGRA8_UNorm_SRGB:
        case GpuFormat::BGRX8_UNorm:
        case GpuFormat::BGRX8_UNorm_SRGB:
            return GpuBaseFormat::UNorm;

        case GpuFormat::R16_Uint:
        case GpuFormat::RG16_Uint:
        case GpuFormat::RGBA16_Uint:
        case GpuFormat::R32_Uint:
        case GpuFormat::RG32_Uint:
        case GpuFormat::RGB32_Uint:
        case GpuFormat::RGBA32_Uint:
            return GpuBaseFormat::Uint;

        case GpuFormat::R16_Sint:
        case GpuFormat::RG16_Sint:
        case GpuFormat::RGBA16_Sint:
        case GpuFormat::R32_Sint:
        case GpuFormat::RG32_Sint:
        case GpuFormat::RGB32_Sint:
        case GpuFormat::RGBA32_Sint:
            return GpuBaseFormat::Sint;

        case GpuFormat::R16_Float:
        case GpuFormat::RG16_Float:
        case GpuFormat::RGBA16_Float:
            return GpuBaseFormat::Float;

        case GpuFormat::R32_Float:
        case GpuFormat::RG32_Float:
        case GpuFormat::RGB32_Float:
        case GpuFormat::RGBA32_Float:
            return GpuBaseFormat::Float;

        default:
            Unreachable("Invalid format");
        }
    }

    uint32_t FormatChannels(GpuFormat fmt)
    {
        switch (fmt)
        {
        case GpuFormat::Unknown:
            return 0;

        case GpuFormat::R8_UNorm:
        case GpuFormat::R16_Uint:
        case GpuFormat::R16_Sint:
        case GpuFormat::R16_Float:
        case GpuFormat::D16_UNorm:
        case GpuFormat::R32_Uint:
        case GpuFormat::R32_Sint:
        case GpuFormat::R32_Float:
        case GpuFormat::D32_Float:
            return 1;

        case GpuFormat::RG8_UNorm:
        case GpuFormat::RG16_Uint:
        case GpuFormat::RG16_Sint:
        case GpuFormat::RG16_Float:
        case GpuFormat::D24_UNorm_S8_Uint:
        case GpuFormat::RG32_Uint:
        case GpuFormat::RG32_Sint:
        case GpuFormat::RG32_Float:
        case GpuFormat::D32_Float_S8X24_Uint:
            return 2;

        case GpuFormat::RGB32_Uint:
        case GpuFormat::RGB32_Sint:
        case GpuFormat::RGB32_Float:
            return 3;

        case GpuFormat::RGBA8_UNorm:
        case GpuFormat::RGBA8_UNorm_SRGB:
        case GpuFormat::BGRA8_UNorm:
        case GpuFormat::BGRA8_UNorm_SRGB:
        case GpuFormat::BGRX8_UNorm:
        case GpuFormat::BGRX8_UNorm_SRGB:
        case GpuFormat::RGBA16_Uint:
        case GpuFormat::RGBA16_Sint:
        case GpuFormat::RGBA16_Float:
        case GpuFormat::RGBA32_Uint:
        case GpuFormat::RGBA32_Sint:
        case GpuFormat::RGBA32_Float:
            return 4;

        default:
            Unreachable("Invalid format");
        }
    }

    uint32_t FormatChannelSize(GpuFormat fmt)
    {
        switch (fmt)
        {
        case GpuFormat::Unknown:
            return 0;

        case GpuFormat::R8_UNorm:
        case GpuFormat::RG8_UNorm:
        case GpuFormat::RGBA8_UNorm:
        case GpuFormat::RGBA8_UNorm_SRGB:
        case GpuFormat::BGRA8_UNorm:
        case GpuFormat::BGRA8_UNorm_SRGB:
        case GpuFormat::BGRX8_UNorm:
        case GpuFormat::BGRX8_UNorm_SRGB:
            return 1;

        case GpuFormat::R16_Uint:
        case GpuFormat::R16_Sint:
        case GpuFormat::R16_Float:
        case GpuFormat::D16_UNorm:
        case GpuFormat::RG16_Uint:
        case GpuFormat::RG16_Sint:
        case GpuFormat::RG16_Float:
        case GpuFormat::RGBA16_Uint:
        case GpuFormat::RGBA16_Sint:
        case GpuFormat::RGBA16_Float:
            return 2;

        case GpuFormat::R32_Uint:
        case GpuFormat::R32_Sint:
        case GpuFormat::R32_Float:
        case GpuFormat::D32_Float:
        case GpuFormat::RG32_Uint:
        case GpuFormat::RG32_Sint:
        case GpuFormat::RG32_Float:
        case GpuFormat::RGB32_Uint:
        case GpuFormat::RGB32_Sint:
        case GpuFormat::RGB32_Float:
        case GpuFormat::RGBA32_Uint:
        case GpuFormat::RGBA32_Sint:
        case GpuFormat::RGBA32_Float:
            return 4;

        default:
            Unreachable("Invalid format");
        }
    }

    uint32_t NumPlanes(GpuFormat fmt) noexcept
    {
        switch (fmt)
        {
        case GpuFormat::D24_UNorm_S8_Uint:
        case GpuFormat::D32_Float_S8X24_Uint:
        case GpuFormat::NV12:
            // TODO: Support more formats
            return 2;

        default:
            return 1;
        }
    }

    bool IsDepthStencilFormat(GpuFormat fmt) noexcept
    {
        return (fmt == GpuFormat::D16_UNorm) || (fmt == GpuFormat::D24_UNorm_S8_Uint) || (fmt == GpuFormat::D32_Float) ||
               (fmt == GpuFormat::D32_Float_S8X24_Uint);
    }

    bool IsStencilFormat(GpuFormat fmt) noexcept
    {
        return (fmt == GpuFormat::D24_UNorm_S8_Uint) || (fmt == GpuFormat::D32_Float_S8X24_Uint);
    }
} // namespace AIHoloImager
