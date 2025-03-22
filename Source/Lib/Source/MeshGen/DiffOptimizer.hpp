// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>

#include <glm/mat4x4.hpp>

#include "AIHoloImager/Mesh.hpp"
#include "Python/PythonSystem.hpp"
#include "SfM/StructureFromMotion.hpp"
#include "Util/BoundingBox.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class DiffOptimizer
    {
        DISALLOW_COPY_AND_ASSIGN(DiffOptimizer);

    public:
        explicit DiffOptimizer(PythonSystem& python_system);
        DiffOptimizer(DiffOptimizer&& other) noexcept;
        ~DiffOptimizer() noexcept;

        DiffOptimizer& operator=(DiffOptimizer&& other) noexcept;

        void Optimize(Mesh& mesh, glm::mat4x4& model_mtx, const Obb& world_obb, const StructureFromMotion::Result& sfm_input);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
