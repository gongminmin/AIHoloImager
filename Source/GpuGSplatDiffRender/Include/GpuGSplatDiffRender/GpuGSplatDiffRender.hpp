// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <memory>

#include <glm/mat4x4.hpp>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"
#include "GpuGSplatDiffRender/Symbol.hpp"

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

    class GpuGSplatDiffRender
    {
        DISALLOW_COPY_AND_ASSIGN(GpuGSplatDiffRender);

    public:
        AIHI_GPU_GSPLAT_DIFF_RENDER_API GpuGSplatDiffRender() noexcept;
        AIHI_GPU_GSPLAT_DIFF_RENDER_API explicit GpuGSplatDiffRender(GpuSystem& gpu_system);
        AIHI_GPU_GSPLAT_DIFF_RENDER_API GpuGSplatDiffRender(GpuGSplatDiffRender&& other) noexcept;
        AIHI_GPU_GSPLAT_DIFF_RENDER_API ~GpuGSplatDiffRender() noexcept;

        AIHI_GPU_GSPLAT_DIFF_RENDER_API GpuGSplatDiffRender& operator=(GpuGSplatDiffRender&& other) noexcept;

        AIHI_GPU_GSPLAT_DIFF_RENDER_API void Render(GpuCommandList& cmd_list, const Gaussians& gaussians, const glm::mat4x4& model_mtx,
            const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, const GpuViewport& viewport, float kernel_size,
            GpuTexture2D& rendered_image);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    constexpr uint32_t NumCoefficientsFromShDegrees(uint32_t degrees) noexcept
    {
        return (degrees + 1) * (degrees + 1);
    }
} // namespace AIHoloImager
