// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <memory>

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "MeshGen/MeshGenerator.hpp"
#include "SfM/StructureFromMotion.hpp"
#include "Util/GpuMesh.hpp"

namespace AIHoloImager
{
    class MeshOptimizer
    {
        DISALLOW_COPY_AND_ASSIGN(MeshOptimizer);

    public:
        explicit MeshOptimizer(AIHoloImagerInternal& aihi);
        MeshOptimizer(MeshOptimizer&& other) noexcept;
        ~MeshOptimizer() noexcept;

        MeshOptimizer& operator=(MeshOptimizer&& other) noexcept;

        GpuMesh Optimize(const StructureFromMotion::Result& sfm_input, const MeshGenerator::Result& mg_input, uint32_t texture_size);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
