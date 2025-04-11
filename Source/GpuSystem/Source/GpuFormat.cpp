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

    DXGI_FORMAT ToDxgiFormat(GpuFormat fmt)
    {
        switch (fmt)
        {
        case GpuFormat::Unknown:
            return DXGI_FORMAT_UNKNOWN;

        case GpuFormat::R8_UNorm:
            return DXGI_FORMAT_R8_UNORM;

        case GpuFormat::RG8_UNorm:
            return DXGI_FORMAT_R8G8_UNORM;

        case GpuFormat::RGBA8_UNorm:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case GpuFormat::RGBA8_UNorm_SRGB:
            return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case GpuFormat::BGRA8_UNorm:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case GpuFormat::BGRA8_UNorm_SRGB:
            return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        case GpuFormat::BGRX8_UNorm:
            return DXGI_FORMAT_B8G8R8X8_UNORM;
        case GpuFormat::BGRX8_UNorm_SRGB:
            return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;

        case GpuFormat::R16_Uint:
            return DXGI_FORMAT_R16_UINT;
        case GpuFormat::R16_Sint:
            return DXGI_FORMAT_R16_SINT;
        case GpuFormat::R16_Float:
            return DXGI_FORMAT_R16_FLOAT;

        case GpuFormat::RG16_Uint:
            return DXGI_FORMAT_R16G16_UINT;
        case GpuFormat::RG16_Sint:
            return DXGI_FORMAT_R16G16_SINT;
        case GpuFormat::RG16_Float:
            return DXGI_FORMAT_R16G16_FLOAT;

        case GpuFormat::RGBA16_Uint:
            return DXGI_FORMAT_R16G16B16A16_UINT;
        case GpuFormat::RGBA16_Sint:
            return DXGI_FORMAT_R16G16B16A16_SINT;
        case GpuFormat::RGBA16_Float:
            return DXGI_FORMAT_R16G16B16A16_FLOAT;

        case GpuFormat::R32_Uint:
            return DXGI_FORMAT_R32_UINT;
        case GpuFormat::R32_Sint:
            return DXGI_FORMAT_R32_SINT;
        case GpuFormat::R32_Float:
            return DXGI_FORMAT_R32_FLOAT;

        case GpuFormat::RG32_Uint:
            return DXGI_FORMAT_R32G32_UINT;
        case GpuFormat::RG32_Sint:
            return DXGI_FORMAT_R32G32_SINT;
        case GpuFormat::RG32_Float:
            return DXGI_FORMAT_R32G32_FLOAT;

        case GpuFormat::RGB32_Uint:
            return DXGI_FORMAT_R32G32B32_UINT;
        case GpuFormat::RGB32_Sint:
            return DXGI_FORMAT_R32G32B32_SINT;
        case GpuFormat::RGB32_Float:
            return DXGI_FORMAT_R32G32B32_FLOAT;

        case GpuFormat::RGBA32_Uint:
            return DXGI_FORMAT_R32G32B32A32_UINT;
        case GpuFormat::RGBA32_Sint:
            return DXGI_FORMAT_R32G32B32A32_SINT;
        case GpuFormat::RGBA32_Float:
            return DXGI_FORMAT_R32G32B32A32_FLOAT;

        case GpuFormat::D16_UNorm:
            return DXGI_FORMAT_D16_UNORM;
        case GpuFormat::D24_UNorm_S8_Uint:
            return DXGI_FORMAT_D24_UNORM_S8_UINT;
        case GpuFormat::D32_Float:
            return DXGI_FORMAT_D32_FLOAT;
        case GpuFormat::D32_Float_S8X24_Uint:
            return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;

        default:
            Unreachable("Invalid format");
        }
    }
} // namespace AIHoloImager
