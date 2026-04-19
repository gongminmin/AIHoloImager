// Copyright (c) 2026 Minmin Gong
//

#include "CameraUtil.hpp"

#include <cmath>
#include <numbers>

#include <glm/vec2.hpp>

namespace AIHoloImager
{
    float RadicalInverse(uint32_t n, uint32_t base)
    {
        float inv_base = 1.0f / base;
        float f = inv_base;
        float result = 0;
        while (n > 0)
        {
            result += f * (n % base);
            n /= base;
            f *= inv_base;
        }
        return result;
    }

    float HaltonSequence(uint32_t index)
    {
        return RadicalInverse(index, 2);
    }

    glm::vec2 HammersleySequence(uint32_t index, uint32_t num_samples)
    {
        return glm::vec2(static_cast<float>(index) / num_samples, HaltonSequence(index));
    }

    glm::vec2 SphereHammersleySequence(uint32_t index, uint32_t num_samples)
    {
        const glm::vec2 uv = HammersleySequence(index, num_samples);

        const float theta = std::acos(1 - 2 * uv.x) - std::numbers::pi_v<float> / 2;
        const float phi = uv.y * 2 * std::numbers::pi_v<float>;
        return glm::vec2(phi, theta);
    }

    glm::vec3 SphericalCameraPose(float azimuth, float elevation, float radius)
    {
        const float sin_azimuth = std::sin(azimuth);
        const float cos_azimuth = std::cos(azimuth);

        const float sin_elevation = std::sin(elevation);
        const float cos_elevation = std::cos(elevation);

        const float x = cos_elevation * cos_azimuth;
        const float y = sin_elevation;
        const float z = cos_elevation * sin_azimuth;
        return glm::vec3(x, y, z) * radius;
    }
} // namespace AIHoloImager
