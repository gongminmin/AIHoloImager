// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <bit>

constexpr uint32_t LogNextPowerOf2(uint32_t n)
{
    if (n <= 1)
    {
        return 1;
    }
    return sizeof(uint32_t) * 8 - std::countl_zero(n - 1);
}
