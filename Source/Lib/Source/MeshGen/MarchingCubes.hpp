// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <span>

#include <DirectXMath.h>

#include "AIHoloImager/Mesh.hpp"
#include "Gpu/GpuSystem.hpp"
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

        Mesh Generate(std::span<const DirectX::XMFLOAT4> scalar_deformation, uint32_t grid_res, float isovalue);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
