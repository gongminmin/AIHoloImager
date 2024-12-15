// Copyright (c) 2024 Minmin Gong
//

#include "Util/FormatConversion.hpp"

#include "Util/ErrorHandling.hpp"

namespace AIHoloImager
{
    GpuFormat ToGpuFormat(ElementFormat fmt)
    {
        switch (fmt)
        {
        case ElementFormat::Unknown:
            return GpuFormat::Unknown;

        case ElementFormat::R8_UNorm:
            return GpuFormat::R8_UNorm;
        case ElementFormat::RG8_UNorm:
            return GpuFormat::RG8_UNorm;
        case ElementFormat::RGBA8_UNorm:
            return GpuFormat::RGBA8_UNorm;

        case ElementFormat::R32_Float:
            return GpuFormat::R32_Float;
        case ElementFormat::RG32_Float:
            return GpuFormat::RG32_Float;
        case ElementFormat::RGB32_Float:
            return GpuFormat::RGB32_Float;
        case ElementFormat::RGBA32_Float:
            return GpuFormat::RGBA32_Float;

        default:
            Unreachable("Invalid format");
        }
    }

    ElementFormat ToElementFormat(GpuFormat fmt)
    {
        switch (fmt)
        {
        case GpuFormat::Unknown:
            return ElementFormat::Unknown;

        case GpuFormat::R8_UNorm:
            return ElementFormat::R8_UNorm;
        case GpuFormat::RG8_UNorm:
            return ElementFormat::RG8_UNorm;
        case GpuFormat::RGBA8_UNorm:
            return ElementFormat::RGBA8_UNorm;

        case GpuFormat::R32_Float:
            return ElementFormat::R32_Float;
        case GpuFormat::RG32_Float:
            return ElementFormat::RG32_Float;
        case GpuFormat::RGB32_Float:
            return ElementFormat::RGB32_Float;
        case GpuFormat::RGBA32_Float:
            return ElementFormat::RGBA32_Float;

        default:
            Unreachable("Invalid format");
        }
    }
} // namespace AIHoloImager
