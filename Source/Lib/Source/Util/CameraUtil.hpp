// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace AIHoloImager
{
    glm::vec2 SphereHammersleySequence(uint32_t index, uint32_t num_samples);
    glm::vec3 SphericalCameraPose(float azimuth, float elevation, float radius);
} // namespace AIHoloImager
