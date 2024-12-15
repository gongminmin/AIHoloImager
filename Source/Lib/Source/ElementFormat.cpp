// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/ElementFormat.hpp"

#include "Util/ErrorHandling.hpp"

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
            return 1;

        case ElementFormat::RG8_UNorm:
            return 2;

        case ElementFormat::RGB8_UNorm:
            return 3;

        case ElementFormat::RGBA8_UNorm:
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

        default:
            Unreachable("Invalid format");
        }
    }
} // namespace AIHoloImager
