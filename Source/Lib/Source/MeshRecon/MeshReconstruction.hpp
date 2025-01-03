// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#include <glm/mat4x4.hpp>

#include "AIHoloImager/Mesh.hpp"
#include "Gpu/GpuSystem.hpp"
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
            Mesh mesh;

            glm::mat4x4 transform; // From model space to SfM space
            Obb obb;
        };

    public:
        MeshReconstruction(const std::filesystem::path& exe_dir, GpuSystem& gpu_system);
        MeshReconstruction(MeshReconstruction&& other) noexcept;
        ~MeshReconstruction() noexcept;

        MeshReconstruction& operator=(MeshReconstruction&& other) noexcept;

        Result Process(const StructureFromMotion::Result& sfm_input, uint32_t max_texture_size, const std::filesystem::path& tmp_dir);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    glm::mat4x4 RegularizeTransform(const glm::vec3& translate, const glm::quat& rotation, const glm::vec3& scale);
} // namespace AIHoloImager
