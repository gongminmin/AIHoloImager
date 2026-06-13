// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuTexture.hpp"
#include "Util/GpuMesh.hpp"

namespace AIHoloImager
{
    class MarchingCubes
    {
        DISALLOW_COPY_AND_ASSIGN(MarchingCubes);

    public:
        explicit MarchingCubes(AIHoloImagerInternal& aihi);
        MarchingCubes(MarchingCubes&& other) noexcept;
        ~MarchingCubes() noexcept;

        MarchingCubes& operator=(MarchingCubes&& other) noexcept;

        GpuMesh Generate(const GpuTexture3D& scalar_deformation, float isovalue, float scale);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
