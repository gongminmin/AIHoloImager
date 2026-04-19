// Copyright (c) 2026 Minmin Gong
//

#pragma once

#ifdef AIHI_KEEP_INTERMEDIATES
    #include <filesystem>
#endif
#include <memory>

#include <glm/mat4x4.hpp>

#include "AIHoloImagerInternal.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuBuffer.hpp"
#ifdef AIHI_KEEP_INTERMEDIATES
    #include "Gpu/GpuSystem.hpp"
#endif
#include "Gpu/GpuTexture.hpp"

namespace AIHoloImager
{
    struct Gaussians
    {
        uint32_t num_gaussians;
        uint32_t sh_degrees;

        GpuBuffer positions; // n float3s
        GpuBuffer scales;    // n float3s
        GpuBuffer rotations; // n float4s
        GpuBuffer shs;       // n * (l + 1) ^ 2 float3s
        GpuBuffer opacities; // n floats
    };

    class GaussianSplatting
    {
        DISALLOW_COPY_AND_ASSIGN(GaussianSplatting);

    public:
        GaussianSplatting() noexcept;
        explicit GaussianSplatting(AIHoloImagerInternal& aihi);
        GaussianSplatting(GaussianSplatting&& other) noexcept;
        ~GaussianSplatting() noexcept;

        GaussianSplatting& operator=(GaussianSplatting&& other) noexcept;

        void Render(const Gaussians& gaussians, const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, float kernel_size,
            GpuTexture2D& rendered_image);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

#ifdef AIHI_KEEP_INTERMEDIATES
    void SavePointCloud(GpuSystem& gpu_system, const Gaussians& gaussians, const std::filesystem::path& path);

    Gaussians LoadGaussians(GpuSystem& gpu_system, const std::filesystem::path& path);
    void SaveGaussians(GpuSystem& gpu_system, const Gaussians& gaussians, const std::filesystem::path& path);
#endif
} // namespace AIHoloImager
