// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <span>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace AIHoloImager
{
    struct Obb
    {
        glm::vec3 center;
        glm::vec3 extents;
        glm::quat orientation;

        static Obb FromPoints(const glm::vec3* positions, uint32_t stride, uint32_t num_vertices);
        static Obb Transform(const Obb& obb, const glm::mat4x4& mtx);
        static void GetCorners(const Obb& obb, std::span<glm::vec3> corners);
    };
} // namespace AIHoloImager
