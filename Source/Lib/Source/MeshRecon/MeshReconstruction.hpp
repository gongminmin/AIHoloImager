// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#include <glm/mat4x4.hpp>

#include "SfM/StructureFromMotion.hpp"
#include "Util/BoundingBox.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class MeshReconstruction
    {
        DISALLOW_COPY_AND_ASSIGN(MeshReconstruction);

    public:
        struct Result
        {
            glm::mat4x4 transform; // From model space to SfM space
            Obb obb;
        };

    public:
        explicit MeshReconstruction(const std::filesystem::path& exe_dir);
        MeshReconstruction(MeshReconstruction&& other) noexcept;
        ~MeshReconstruction() noexcept;

        MeshReconstruction& operator=(MeshReconstruction&& other) noexcept;

        Result Process(const StructureFromMotion::Result& sfm_input, const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    glm::mat4x4 RegularizeTransform(const glm::vec3& translate, const glm::quat& rotation, const glm::vec3& scale);
} // namespace AIHoloImager
