// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include <glm/mat3x3.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "AIHoloImager/Texture.hpp"
#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuTexture.hpp"
#include "Util/BoundingBox.hpp"

namespace AIHoloImager
{
    class StructureFromMotion
    {
        DISALLOW_COPY_AND_ASSIGN(StructureFromMotion);

    public:
        struct View
        {
            GpuTexture2D delighted_tex;
            glm::uvec2 delighted_offset;

            uint32_t intrinsic_id;

            glm::dmat3x3 rotation;
            glm::dvec3 center;
        };

        struct PinholeIntrinsic
        {
            uint32_t width;
            uint32_t height;

            glm::dmat3x3 k;
        };

        struct Observation
        {
            uint32_t view_id;
            glm::dvec2 point;
        };

        struct Landmark
        {
            glm::dvec3 point;
            std::vector<Observation> obs;
        };

        struct Result
        {
            std::vector<View> views;
            std::vector<PinholeIntrinsic> intrinsics;

            std::vector<Landmark> structure;
        };

    public:
        explicit StructureFromMotion(AIHoloImagerInternal& aihi);
        StructureFromMotion(StructureFromMotion&& other) noexcept;
        ~StructureFromMotion() noexcept;

        StructureFromMotion& operator=(StructureFromMotion&& other) noexcept;

        Result Process(const std::filesystem::path& input_path, bool sequential);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    glm::mat4x4 CalcViewMatrix(const StructureFromMotion::View& view);
    glm::mat4x4 CalcProjMatrix(const StructureFromMotion::PinholeIntrinsic& intrinsic, float near_plane, float far_plane);
    glm::vec2 CalcNearFarPlane(const glm::mat4x4& view_mtx, const Obb& obb);
    glm::vec2 CalcViewportOffset(const StructureFromMotion::PinholeIntrinsic& intrinsic);
} // namespace AIHoloImager
