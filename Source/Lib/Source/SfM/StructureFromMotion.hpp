// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class StructureFromMotion
    {
        DISALLOW_COPY_AND_ASSIGN(StructureFromMotion);

    public:
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
            std::vector<AIHoloImagerInternal::ProjectionDesc> projections;
            std::vector<Landmark> structure;
        };

    public:
        explicit StructureFromMotion(AIHoloImagerInternal& aihi);
        StructureFromMotion(StructureFromMotion&& other) noexcept;
        ~StructureFromMotion() noexcept;

        StructureFromMotion& operator=(StructureFromMotion&& other) noexcept;

        Result Process(const std::filesystem::path& input_path, bool sequential, bool no_delight);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
