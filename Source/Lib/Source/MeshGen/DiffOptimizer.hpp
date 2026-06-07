// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <glm/mat4x4.hpp>

#include "AIHoloImager/Mesh.hpp"
#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"

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

        void OptimizeTransform(const Mesh& mesh, glm::mat4x4& model_mtx, std::span<const AIHoloImagerInternal::ProjectionDesc> projections);
        void OptimizeTexture(Mesh& mesh, const glm::mat4x4& model_mtx, std::span<const AIHoloImagerInternal::ProjectionDesc> projections,
            GpuTexture2D& albedo_tex, const GpuTexture2D& mask_tex);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
