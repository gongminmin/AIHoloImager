// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include "AIHoloImager/ElementFormat.hpp"
#include "Gpu/GpuFormat.hpp"

namespace AIHoloImager
{
    GpuFormat ToGpuFormat(ElementFormat fmt);
    ElementFormat ToElementFormat(GpuFormat fmt);
} // namespace AIHoloImager
