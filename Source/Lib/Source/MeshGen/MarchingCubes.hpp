// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <span>

#include "AIHoloImager/Mesh.hpp"

namespace AIHoloImager
{
    Mesh MarchingCubes(std::span<const float> sdf, uint32_t grid_res, float isovalue);
} // namespace AIHoloImager
