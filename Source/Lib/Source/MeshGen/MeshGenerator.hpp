// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <glm/vec3.hpp>

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "GSplat/GaussianSplatting.hpp"
#include "SfM/StructureFromMotion.hpp"
#include "Util/BoundingBox.hpp"
#include "Util/GpuMesh.hpp"

namespace AIHoloImager
{
    class MeshGenerator
    {
        DISALLOW_COPY_AND_ASSIGN(MeshGenerator);

    public:
        struct Result
        {
            GpuMesh mesh;
            Gaussians gaussians;
            Aabb obj_aabb;
            glm::vec3 up_vec;
        };

    public:
        explicit MeshGenerator(AIHoloImagerInternal& aihi);
        MeshGenerator(MeshGenerator&& other) noexcept;
        ~MeshGenerator() noexcept;

        MeshGenerator& operator=(MeshGenerator&& other) noexcept;

        Result Generate(const StructureFromMotion::Result& sfm_input);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
