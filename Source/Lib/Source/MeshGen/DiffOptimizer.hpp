// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <glm/mat4x4.hpp>

#include "AIHoloImager/Mesh.hpp"
#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "SfM/StructureFromMotion.hpp"

namespace AIHoloImager
{
    class DiffOptimizer
    {
        DISALLOW_COPY_AND_ASSIGN(DiffOptimizer);

    public:
        explicit DiffOptimizer(AIHoloImagerInternal& aihi);
        DiffOptimizer(DiffOptimizer&& other) noexcept;
        ~DiffOptimizer() noexcept;

        DiffOptimizer& operator=(DiffOptimizer&& other) noexcept;

        void Optimize(Mesh& mesh, glm::mat4x4& model_mtx, const StructureFromMotion::Result& sfm_input);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
