// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <limits>
#include <span>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace AIHoloImager
{
    struct Aabb
    {
        glm::vec3 min{std::numeric_limits<float>::max()};
        glm::vec3 max{std::numeric_limits<float>::lowest()};

        void AddPoint(const glm::vec3& point);
        glm::vec3 Center() const;
        glm::vec3 Extents() const;
        glm::vec3 Size() const;

        static void GetCorners(const Aabb& aabb, std::span<glm::vec3> corners);
    };

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
