// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <bit>
#include <string>
#include <string_view>

namespace AIHoloImager
{
    constexpr uint32_t LogNextPowerOf2(uint32_t n)
    {
        if (n <= 1)
        {
            return 1;
        }
        return sizeof(uint32_t) * 8 - std::countl_zero(n - 1) + 1;
    }

    // The strings here actually store UTF-8 encoded data
    void Convert(std::string& dest, std::string_view src);
    void Convert(std::string& dest, std::u16string_view src);
    void Convert(std::u16string& dest, std::string_view src);
    void Convert(std::u16string& dest, std::u16string_view src);
} // namespace AIHoloImager
