// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include "AIHoloImager/Mesh.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class MarchingCubes
    {
        DISALLOW_COPY_AND_ASSIGN(MarchingCubes);

    public:
        explicit MarchingCubes(GpuSystem& gpu_system);
        MarchingCubes(MarchingCubes&& other) noexcept;
        ~MarchingCubes() noexcept;

        MarchingCubes& operator=(MarchingCubes&& other) noexcept;

        Mesh Generate(const GpuTexture3D& scalar_deformation, float isovalue, float scale);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
