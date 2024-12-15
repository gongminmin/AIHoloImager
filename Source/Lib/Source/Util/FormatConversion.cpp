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

        default:
            Unreachable("Invalid format");
        }
    }
} // namespace AIHoloImager