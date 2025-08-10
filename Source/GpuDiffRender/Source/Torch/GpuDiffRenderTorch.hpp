// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <array>
#include <tuple>

#ifdef _MSC_VER
    #pragma warning(push)
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

#include "../GpuDiffRender.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuTexture.hpp"
#include "TensorConverter/TensorConverter.hpp"

namespace AIHoloImager
{
    class GpuDiffRenderTorch
    {
    public:
        GpuDiffRenderTorch(size_t gpu_system, torch::Device torch_device);
        ~GpuDiffRenderTorch();

        struct Viewport
        {
            float left;
            float top;
            float width;
            float height;
        };

        torch::autograd::tensor_list Rasterize(
            torch::Tensor positions, torch::Tensor indices, const std::array<uint32_t, 2>& resolution, const Viewport* viewport = nullptr);

        torch::Tensor Interpolate(torch::Tensor vtx_attribs, torch::Tensor barycentric, torch::Tensor prim_id, torch::Tensor indices);

        struct AntiAliasOppositeVertices
        {
            GpuBuffer opposite_vertices;
        };

        AntiAliasOppositeVertices AntiAliasConstructOppositeVertices(torch::Tensor indices);

        torch::Tensor AntiAlias(torch::Tensor shading, torch::Tensor prim_id, torch::Tensor positions, torch::Tensor indices,
            const Viewport* viewport = nullptr, const AntiAliasOppositeVertices* opposite_vertices = nullptr);

    private:
        std::tuple<torch::Tensor, torch::Tensor> RasterizeFwd(
            torch::Tensor positions, torch::Tensor indices, const std::array<uint32_t, 2>& resolution, const Viewport* viewport = nullptr);
        torch::Tensor RasterizeBwd(torch::Tensor grad_barycentric);

        torch::Tensor InterpolateFwd(torch::Tensor vtx_attribs, torch::Tensor barycentric, torch::Tensor prim_id, torch::Tensor indices);
        std::tuple<torch::Tensor, torch::Tensor> InterpolateBwd(torch::Tensor grad_shading);

        torch::Tensor AntiAliasFwd(torch::Tensor shading, torch::Tensor prim_id, torch::Tensor positions, torch::Tensor indices,
            const Viewport* viewport = nullptr, const AntiAliasOppositeVertices* opposite_vertices = nullptr);
        std::tuple<torch::Tensor, torch::Tensor> AntiAliasBwd(torch::Tensor grad_anti_aliased);

    private:
        GpuSystem& gpu_system_;
        torch::Device torch_device_;

        GpuDiffRender gpu_dr_;

        TensorConverter tensor_converter_;

        struct RasterizeIntermediate
        {
            GpuBuffer positions;
            GpuBuffer indices;
            GpuTexture2D barycentric;
            GpuTexture2D prim_id;
            GpuViewport viewport;

            GpuTexture2D grad_barycentric;
            GpuBuffer grad_positions;
        };
        RasterizeIntermediate rast_intermediate_;

        struct InterpolateIntermediate
        {
            GpuBuffer vtx_attribs;
            uint32_t num_attribs;
            GpuTexture2D barycentric;
            GpuTexture2D prim_id;
            GpuBuffer indices;

            GpuBuffer shading;
            GpuBuffer grad_shading;
            GpuBuffer grad_vtx_attribs;
            GpuTexture2D grad_barycentric;
        };
        InterpolateIntermediate interpolate_intermediate_;

        struct AntiAliasIntermediate
        {
            GpuBuffer shading;
            GpuTexture2D prim_id;
            GpuBuffer positions;
            GpuBuffer indices;
            GpuViewport viewport;

            GpuBuffer anti_aliased;
            GpuBuffer grad_anti_aliased;
            GpuBuffer grad_shading;
            GpuBuffer grad_positions;
        };
        AntiAliasIntermediate aa_intermediate_;
    };
} // namespace AIHoloImager
