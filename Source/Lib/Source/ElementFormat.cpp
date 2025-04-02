// Copyright (c) 2024-2025 Minmin Gong
//

#include "AIHoloImager/ElementFormat.hpp"

#include "Base/ErrorHandling.hpp"

namespace AIHoloImager
{
    uint32_t FormatSize(ElementFormat fmt)
    {
        return FormatChannelSize(fmt) * FormatChannels(fmt);
    }

    uint32_t FormatChannels(ElementFormat fmt)
    {
        switch (fmt)
        {
        case ElementFormat::Unknown:
            return 0;

        case ElementFormat::R8_UNorm:
        case ElementFormat::R32_Float:
            return 1;

        case ElementFormat::RG8_UNorm:
        case ElementFormat::RG32_Float:
            return 2;

        case ElementFormat::RGB8_UNorm:
        case ElementFormat::RGB32_Float:
            return 3;

        case ElementFormat::RGBA8_UNorm:
        case ElementFormat::RGBA32_Float:
            return 4;

        default:
            Unreachable("Invalid format");
        }
    }

    uint32_t FormatChannelSize(ElementFormat fmt)
    {
        switch (fmt)
        {
        case ElementFormat::Unknown:
            return 0;

        case ElementFormat::R8_UNorm:
        case ElementFormat::RG8_UNorm:
        case ElementFormat::RGB8_UNorm:
        case ElementFormat::RGBA8_UNorm:
            return 1;

        case ElementFormat::R32_Float:
        case ElementFormat::RG32_Float:
        case ElementFormat::RGB32_Float:
        case ElementFormat::RGBA32_Float:
            return 4;

        default:
            Unreachable("Invalid format");
        }
    }
} // namespace AIHoloImager
