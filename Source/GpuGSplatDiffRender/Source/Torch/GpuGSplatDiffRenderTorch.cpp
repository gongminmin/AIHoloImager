// Copyright (c) 2026 Minmin Gong
//

#include "GpuGSplatDiffRenderTorch.hpp"

#include <glm/gtc/type_ptr.hpp>

using namespace torch::autograd;

namespace AIHoloImager
{
    GpuGSplatDiffRenderTorch::GpuGSplatDiffRenderTorch(size_t gpu_system, torch::Device torch_device)
        : gpu_system_(*reinterpret_cast<GpuSystem*>(gpu_system)), torch_device_(std::move(torch_device)), gpu_dr_(gpu_system_),
          tensor_converter_(gpu_system_, torch_device_)
    {
    }

    GpuGSplatDiffRenderTorch::~GpuGSplatDiffRenderTorch() = default;

    torch::Tensor GpuGSplatDiffRenderTorch::Render(torch::Tensor positions, torch::Tensor scales, torch::Tensor rotations,
        torch::Tensor shs, torch::Tensor opacities, const RenderSettings& render_settings)
    {
        struct RenderAutogradFunc : public Function<RenderAutogradFunc>
        {
            static torch::Tensor forward(AutogradContext* ctx, GpuGSplatDiffRenderTorch* dr, torch::Tensor positions, torch::Tensor scales,
                torch::Tensor rotations, torch::Tensor shs, torch::Tensor opacities, const RenderSettings& render_settings)
            {
                auto color = dr->RenderFwd(positions, scales, rotations, shs, opacities, render_settings);

                ctx->saved_data["render_settings"] = reinterpret_cast<int64_t>(&render_settings);
                ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);

                return color;
            }

            static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
            {
                auto* dr = reinterpret_cast<GpuGSplatDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());
                const auto& render_settings = *reinterpret_cast<const RenderSettings*>(ctx->saved_data["render_settings"].to<int64_t>());
                auto saved = ctx->get_saved_variables();

                torch::Tensor grad_image = std::move(grad_outputs[0]);

                auto [grad_positions, grad_scales, grad_rotations, grad_shs, grad_opacities] = dr->RenderBwd(grad_image, render_settings);

                return {torch::Tensor(), std::move(grad_positions), std::move(grad_scales), std::move(grad_rotations), std::move(grad_shs),
                    std::move(grad_opacities), torch::Tensor()};
            }
        };

        return RenderAutogradFunc::apply(
            this, std::move(positions), std::move(scales), std::move(rotations), std::move(shs), std::move(opacities), render_settings);
    }

    torch::Tensor GpuGSplatDiffRenderTorch::RenderFwd(torch::Tensor positions, torch::Tensor scales, torch::Tensor rotations,
        torch::Tensor shs, torch::Tensor opacities, const RenderSettings& render_settings)
    {
        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        render_intermediate_.gaussians.num_gaussians = static_cast<uint32_t>(positions.size(0));
        render_intermediate_.gaussians.sh_degrees = render_settings.sh_degree;

        tensor_converter_.Convert(cmd_list, std::move(positions), render_intermediate_.gaussians.positions, GpuHeap::Default,
            GpuResourceFlag::VertexBuffer | GpuResourceFlag::ShaderResource, "GpuGSplatDiffRenderTorch.RenderFwd.positions");
        tensor_converter_.Convert(cmd_list, std::move(scales), render_intermediate_.gaussians.scales, GpuHeap::Default,
            GpuResourceFlag::VertexBuffer | GpuResourceFlag::ShaderResource, "GpuGSplatDiffRenderTorch.RenderFwd.scales");
        tensor_converter_.Convert(cmd_list, std::move(rotations), render_intermediate_.gaussians.rotations, GpuHeap::Default,
            GpuResourceFlag::VertexBuffer | GpuResourceFlag::ShaderResource, "GpuGSplatDiffRenderTorch.RenderFwd.rotations");
        tensor_converter_.Convert(cmd_list, std::move(shs), render_intermediate_.gaussians.shs, GpuHeap::Default,
            GpuResourceFlag::VertexBuffer | GpuResourceFlag::ShaderResource, "GpuGSplatDiffRenderTorch.RenderFwd.shs");
        tensor_converter_.Convert(cmd_list, std::move(opacities), render_intermediate_.gaussians.opacities, GpuHeap::Default,
            GpuResourceFlag::VertexBuffer | GpuResourceFlag::ShaderResource, "GpuGSplatDiffRenderTorch.RenderFwd.opacities");

        if (render_settings.viewport != nullptr)
        {
            render_intermediate_.viewport = {
                .left = render_settings.viewport->left,
                .top = render_settings.viewport->top,
                .width = render_settings.viewport->width,
                .height = render_settings.viewport->height,
            };
        }
        else
        {
            render_intermediate_.viewport = {
                .left = 0,
                .top = 0,
                .width = static_cast<float>(render_settings.image_width),
                .height = static_cast<float>(render_settings.image_height),
            };
        }
        render_intermediate_.viewport.min_depth = 0;
        render_intermediate_.viewport.max_depth = 1;

        render_intermediate_.rendered_buff =
            GpuBuffer(gpu_system_, render_settings.image_width * render_settings.image_height * sizeof(glm::vec3), GpuHeap::Default,
                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable,
                "gsplat.render_intermediate_.rendered_buff");

        gpu_dr_.RenderFwd(cmd_list, render_intermediate_.gaussians, glm::make_mat4(render_settings.model_mtx.data()),
            glm::make_mat4(render_settings.view_mtx.data()), glm::make_mat4(render_settings.proj_mtx.data()), render_intermediate_.viewport,
            render_settings.kernel_size, render_intermediate_.rendered_buff, render_settings.image_width, render_settings.image_height,
            glm::make_vec3(render_settings.bg_color.data()));

        torch::Tensor rendered_image = tensor_converter_.Convert(cmd_list, render_intermediate_.rendered_buff,
            {{render_settings.image_height, render_settings.image_width, 3}}, TensorConverter::DataType::Float32);

        gpu_system_.Execute(std::move(cmd_list));

        return rendered_image;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> GpuGSplatDiffRenderTorch::RenderBwd(
        torch::Tensor grad_image, const RenderSettings& render_settings)
    {
        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(cmd_list, std::move(grad_image), render_intermediate_.grad_image, GpuHeap::Default,
            GpuResourceFlag::ShaderResource, "GpuGSplatDiffRenderTorch.RenderBwd.grad_image");

        gpu_dr_.RenderBwd(cmd_list, render_intermediate_.gaussians, glm::make_mat4(render_settings.model_mtx.data()),
            glm::make_mat4(render_settings.view_mtx.data()), glm::make_mat4(render_settings.proj_mtx.data()), render_intermediate_.viewport,
            render_settings.kernel_size, render_intermediate_.grad_image, render_intermediate_.grad_positions,
            render_intermediate_.grad_scales, render_intermediate_.grad_rotations, render_intermediate_.grad_shs,
            render_intermediate_.grad_opacities);

        torch::Tensor grad_positions = tensor_converter_.Convert(cmd_list, render_intermediate_.grad_positions,
            {{render_intermediate_.gaussians.num_gaussians, 3}}, TensorConverter::DataType::Float32);
        torch::Tensor grad_scales = tensor_converter_.Convert(cmd_list, render_intermediate_.grad_scales,
            {{render_intermediate_.gaussians.num_gaussians, 3}}, TensorConverter::DataType::Float32);
        torch::Tensor grad_rotations = tensor_converter_.Convert(cmd_list, render_intermediate_.grad_rotations,
            {{render_intermediate_.gaussians.num_gaussians, 4}}, TensorConverter::DataType::Float32);
        torch::Tensor grad_shs = tensor_converter_.Convert(cmd_list, render_intermediate_.grad_shs,
            {{render_intermediate_.gaussians.num_gaussians, NumCoefficientsFromShDegrees(render_intermediate_.gaussians.sh_degrees), 3}},
            TensorConverter::DataType::Float32);
        torch::Tensor grad_opacities = tensor_converter_.Convert(cmd_list, render_intermediate_.grad_opacities,
            {{render_intermediate_.gaussians.num_gaussians, 1}}, TensorConverter::DataType::Float32);

        gpu_system_.Execute(std::move(cmd_list));

        return {
            std::move(grad_positions), std::move(grad_scales), std::move(grad_rotations), std::move(grad_shs), std::move(grad_opacities)};
    }
} // namespace AIHoloImager
