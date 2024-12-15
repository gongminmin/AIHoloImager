// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>

namespace AIHoloImager
{
    enum class ElementFormat
    {
        Unknown,

        R8_UNorm,
        RG8_UNorm,
        RGB8_UNorm,
        RGBA8_UNorm,

        R32_Float,
        RG32_Float,
        RGB32_Float,
        RGBA32_Float,
    };

    uint32_t FormatSize(ElementFormat fmt);
    uint32_t FormatChannels(ElementFormat fmt);
    uint32_t FormatChannelSize(ElementFormat fmt);
} // namespace AIHoloImager
