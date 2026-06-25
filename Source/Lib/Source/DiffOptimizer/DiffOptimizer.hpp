// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <glm/mat4x4.hpp>

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "Util/GpuMesh.hpp"

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

        void OptimizeTransform(const GpuMesh& mesh, glm::mat4x4& model_mtx,
            std::span<const AIHoloImagerInternal::ProjectionDesc> projections, bool uniform_scaling);
        void OptimizeTexture(GpuMesh& mesh, const glm::mat4x4& model_mtx, std::span<const AIHoloImagerInternal::ProjectionDesc> projections,
            const GpuTexture2D& mask_tex);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
