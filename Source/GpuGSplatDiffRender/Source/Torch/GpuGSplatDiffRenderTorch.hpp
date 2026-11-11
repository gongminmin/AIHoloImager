// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <array>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4018) // Ignore signed/unsigned compare mismatch
    #pragma warning(disable : 4100) // Ignore unreferenced formal parameters
    #pragma warning(disable : 4127) // Ignore constant conditional expression
    #pragma warning(disable : 4244) // Ignore type conversion from `int` to `float`
    #pragma warning(disable : 4251) // Ignore non dll-interface as member
    #pragma warning(disable : 4267) // Ignore type conversion from `size_t` to something else
    #pragma warning(disable : 4324) // Ignore padded structure
    #pragma warning(disable : 4275) // Ignore non dll-interface base class
#endif
#include <torch/autograd.h>
#include <torch/types.h>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#include "GpuGSplatDiffRender/GpuGSplatDiffRender.hpp"
#include "TensorConverter/TensorConverter.hpp"

namespace AIHoloImager
{
    class GpuGSplatDiffRenderTorch
    {
    public:
        GpuGSplatDiffRenderTorch(size_t gpu_system, torch::Device torch_device);
        ~GpuGSplatDiffRenderTorch();

        struct Viewport
        {
            float left;
            float top;
            float width;
            float height;
        };

        struct RenderSettings
        {
            uint32_t image_width;
            uint32_t image_height;
            uint32_t sh_degree;
            std::array<float, 16> model_mtx;
            std::array<float, 16> view_mtx;
            std::array<float, 16> proj_mtx;
            Viewport* viewport;
            float kernel_size;
            std::array<float, 3> bg_color;
        };

        torch::Tensor Render(torch::Tensor positions, torch::Tensor scales, torch::Tensor rotations, torch::Tensor shs,
            torch::Tensor opacities, const RenderSettings& render_settings);

    private:
        torch::Tensor RenderFwd(torch::Tensor positions, torch::Tensor scales, torch::Tensor rotations, torch::Tensor shs,
            torch::Tensor opacities, const RenderSettings& render_settings);
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RenderBwd(
            torch::Tensor grad_image, const RenderSettings& render_settings);

    private:
        GpuSystem& gpu_system_;
        torch::Device torch_device_;

        GpuGSplatDiffRender gpu_dr_;

        TensorConverter tensor_converter_;

        struct RenderIntermediate
        {
            Gaussians gaussians;
            GpuViewport viewport;
            GpuTexture2D rendered_tex;
            GpuBuffer rendered_buff;

            GpuBuffer grad_image;
            GpuBuffer grad_positions;
            GpuBuffer grad_scales;
            GpuBuffer grad_rotations;
            GpuBuffer grad_shs;
            GpuBuffer grad_opacities;
        };
        RenderIntermediate render_intermediate_;
    };
} // namespace AIHoloImager
