// Copyright (c) 2025 Minmin Gong
//

#include "GpuDiffRenderTorch.hpp"

#include <glm/vec4.hpp>

using namespace torch::autograd;

namespace AIHoloImager
{
    GpuDiffRenderTorch::GpuDiffRenderTorch(size_t gpu_system, torch::Device torch_device)
        : gpu_system_(*reinterpret_cast<GpuSystem*>(gpu_system)), torch_device_(std::move(torch_device)), gpu_dr_(gpu_system_),
          tensor_converter_(gpu_system_, torch_device_)
    {
    }

    GpuDiffRenderTorch::~GpuDiffRenderTorch() = default;

    tensor_list GpuDiffRenderTorch::Rasterize(
        torch::Tensor positions, torch::Tensor indices, const std::array<uint32_t, 2>& resolution, const Viewport* viewport)
    {
        struct RasterizeAutogradFunc : public Function<RasterizeAutogradFunc>
        {
            static tensor_list forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor positions, torch::Tensor indices,
                const std::array<uint32_t, 2>& resolution, const Viewport* viewport)
            {
                auto [barycentric, prim_id] = dr->RasterizeFwd(std::move(positions), std::move(indices), resolution, viewport);
                ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                return {std::move(barycentric), std::move(prim_id)};
            }

            static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
            {
                auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                torch::Tensor grad_barycentric = std::move(grad_outputs[0]);
                auto grad_positions = dr->RasterizeBwd(std::move(grad_barycentric));
                return {torch::Tensor(), std::move(grad_positions), torch::Tensor(), torch::Tensor(), torch::Tensor()};
            }
        };

        return RasterizeAutogradFunc::apply(this, std::move(positions), std::move(indices), resolution, viewport);
    }

    std::tuple<torch::Tensor, torch::Tensor> GpuDiffRenderTorch::RasterizeFwd(
        torch::Tensor positions, torch::Tensor indices, const std::array<uint32_t, 2>& resolution, const Viewport* viewport)
    {
        const uint32_t width = resolution[0];
        const uint32_t height = resolution[1];

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(cmd_list, std::move(positions), rast_intermediate_.positions, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.RasterizeFwd.positions");
        tensor_converter_.Convert(cmd_list, std::move(indices), rast_intermediate_.indices, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.RasterizeFwd.indices");

        if (viewport != nullptr)
        {
            rast_intermediate_.viewport.left = viewport->left;
            rast_intermediate_.viewport.top = viewport->top;
            rast_intermediate_.viewport.width = viewport->width;
            rast_intermediate_.viewport.height = viewport->height;
        }
        else
        {
            rast_intermediate_.viewport.left = 0;
            rast_intermediate_.viewport.top = 0;
            rast_intermediate_.viewport.width = static_cast<float>(width);
            rast_intermediate_.viewport.height = static_cast<float>(height);
        }
        rast_intermediate_.viewport.min_depth = 0;
        rast_intermediate_.viewport.max_depth = 1;

        gpu_dr_.RasterizeFwd(cmd_list, rast_intermediate_.positions, rast_intermediate_.indices, width, height, rast_intermediate_.viewport,
            rast_intermediate_.barycentric, rast_intermediate_.prim_id);

        torch::Tensor barycentric = tensor_converter_.Convert(cmd_list, rast_intermediate_.barycentric);
        torch::Tensor prim_id = tensor_converter_.Convert(cmd_list, rast_intermediate_.prim_id);

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(barycentric), std::move(prim_id)};
    }

    torch::Tensor GpuDiffRenderTorch::RasterizeBwd(torch::Tensor grad_barycentric)
    {
        const uint32_t num_vertices = rast_intermediate_.positions.Size() / sizeof(glm::vec4);

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(cmd_list, std::move(grad_barycentric), rast_intermediate_.grad_barycentric, GpuFormat::RG32_Float,
            GpuResourceFlag::None, L"GpuDiffRenderTorch.RasterizeBwd.grad_barycentric");

        gpu_dr_.RasterizeBwd(cmd_list, rast_intermediate_.positions, rast_intermediate_.indices, rast_intermediate_.viewport,
            rast_intermediate_.barycentric, rast_intermediate_.prim_id, rast_intermediate_.grad_barycentric,
            rast_intermediate_.grad_positions);

        const torch::Tensor grad_positions =
            tensor_converter_.Convert(cmd_list, rast_intermediate_.grad_positions, {num_vertices, 4}, torch::kFloat32);

        gpu_system_.Execute(std::move(cmd_list));

        return grad_positions;
    }

    torch::Tensor GpuDiffRenderTorch::Interpolate(
        torch::Tensor vtx_attribs, torch::Tensor barycentric, torch::Tensor prim_id, torch::Tensor indices)
    {
        struct InterpolateAutogradFunc : public Function<InterpolateAutogradFunc>
        {
            static torch::Tensor forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor vtx_attribs, torch::Tensor barycentric,
                torch::Tensor prim_id, torch::Tensor indices)
            {
                auto shading = dr->InterpolateFwd(std::move(vtx_attribs), std::move(barycentric), std::move(prim_id), std::move(indices));
                ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                return shading;
            }

            static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
            {
                auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                torch::Tensor grad_shading = std::move(grad_outputs[0]);
                auto [grad_vtx_attribs, grad_barycentric] = dr->InterpolateBwd(std::move(grad_shading));
                return {torch::Tensor(), std::move(grad_vtx_attribs), std::move(grad_barycentric), torch::Tensor(), torch::Tensor()};
            }
        };

        return InterpolateAutogradFunc::apply(this, std::move(vtx_attribs), std::move(barycentric), std::move(prim_id), std::move(indices));
    }

    torch::Tensor GpuDiffRenderTorch::InterpolateFwd(
        torch::Tensor vtx_attribs, torch::Tensor barycentric, torch::Tensor prim_id, torch::Tensor indices)
    {
        const uint32_t mini_batch = static_cast<uint32_t>(barycentric.size(0));
        const uint32_t width = static_cast<uint32_t>(barycentric.size(2));
        const uint32_t height = static_cast<uint32_t>(barycentric.size(1));
        const uint32_t num_attribs = static_cast<uint32_t>(vtx_attribs.size(1));

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        interpolate_intermediate_.num_attribs = num_attribs;
        tensor_converter_.Convert(cmd_list, std::move(vtx_attribs), interpolate_intermediate_.vtx_attribs, GpuHeap::Default,
            GpuResourceFlag::None, L"GpuDiffRenderTorch.InterpolateFwd.vtx_attribs");
        tensor_converter_.Convert(cmd_list, std::move(barycentric), interpolate_intermediate_.barycentric, GpuFormat::RG32_Float,
            GpuResourceFlag::None, L"GpuDiffRenderTorch.InterpolateFwd.barycentric");
        tensor_converter_.Convert(cmd_list, std::move(prim_id), interpolate_intermediate_.prim_id, GpuFormat::R32_Uint,
            GpuResourceFlag::None, L"GpuDiffRenderTorch.InterpolateFwd.prim_id");
        tensor_converter_.Convert(cmd_list, std::move(indices), interpolate_intermediate_.indices, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.InterpolateFwd.indices");

        gpu_dr_.InterpolateFwd(cmd_list, interpolate_intermediate_.vtx_attribs, num_attribs, interpolate_intermediate_.barycentric,
            interpolate_intermediate_.prim_id, interpolate_intermediate_.indices, interpolate_intermediate_.shading);

        const torch::Tensor shading = tensor_converter_.Convert(
            cmd_list, interpolate_intermediate_.shading, {mini_batch, height, width, num_attribs}, torch::kFloat32);

        gpu_system_.Execute(std::move(cmd_list));

        return shading;
    }

    std::tuple<torch::Tensor, torch::Tensor> GpuDiffRenderTorch::InterpolateBwd(torch::Tensor grad_shading)
    {
        const uint32_t num_attribs = interpolate_intermediate_.num_attribs;
        const uint32_t num_vertices = interpolate_intermediate_.vtx_attribs.Size() / (num_attribs * sizeof(float));

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(cmd_list, std::move(grad_shading), interpolate_intermediate_.grad_shading, GpuHeap::Default,
            GpuResourceFlag::None, L"GpuDiffRenderTorch.InterpolateBwd.grad_shading");

        gpu_dr_.InterpolateBwd(cmd_list, interpolate_intermediate_.vtx_attribs, num_attribs, interpolate_intermediate_.barycentric,
            interpolate_intermediate_.prim_id, interpolate_intermediate_.indices, interpolate_intermediate_.grad_shading,
            interpolate_intermediate_.grad_vtx_attribs, interpolate_intermediate_.grad_barycentric);

        torch::Tensor grad_vtx_attribs =
            tensor_converter_.Convert(cmd_list, interpolate_intermediate_.grad_vtx_attribs, {num_vertices, num_attribs}, torch::kFloat32);
        torch::Tensor grad_barycentric = tensor_converter_.Convert(cmd_list, interpolate_intermediate_.grad_barycentric);

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(grad_vtx_attribs), std::move(grad_barycentric)};
    }

    GpuDiffRenderTorch::AntiAliasOppositeVertices GpuDiffRenderTorch::AntiAliasConstructOppositeVertices(torch::Tensor indices)
    {
        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        GpuBuffer indices_buff;
        tensor_converter_.Convert(cmd_list, std::move(indices), indices_buff, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasConstructOppositeVertices.indices");

        AntiAliasOppositeVertices ret;
        gpu_dr_.AntiAliasConstructOppositeVertices(cmd_list, indices_buff, ret.opposite_vertices);

        gpu_system_.Execute(std::move(cmd_list));

        return ret;
    }

    torch::Tensor GpuDiffRenderTorch::AntiAlias(torch::Tensor shading, torch::Tensor prim_id, torch::Tensor positions,
        torch::Tensor indices, const Viewport* viewport, const AntiAliasOppositeVertices* opposite_vertices)
    {
        struct AntiAliasAutogradFunc : public Function<AntiAliasAutogradFunc>
        {
            static torch::Tensor forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor shading, torch::Tensor prim_id,
                torch::Tensor positions, torch::Tensor indices, const Viewport* viewport,
                const AntiAliasOppositeVertices* opposite_vertices)
            {
                auto anti_aliased = dr->AntiAliasFwd(
                    std::move(shading), std::move(prim_id), std::move(positions), std::move(indices), viewport, opposite_vertices);
                ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                return anti_aliased;
            }

            static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
            {
                auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                torch::Tensor grad_anti_aliased = std::move(grad_outputs[0]);
                auto [grad_shading, grad_positions] = dr->AntiAliasBwd(std::move(grad_anti_aliased));
                return {torch::Tensor(), std::move(grad_shading), torch::Tensor(), std::move(grad_positions), torch::Tensor(),
                    torch::Tensor(), torch::Tensor()};
            }
        };

        return AntiAliasAutogradFunc::apply(
            this, std::move(shading), std::move(prim_id), std::move(positions), std::move(indices), viewport, opposite_vertices);
    }

    torch::Tensor GpuDiffRenderTorch::AntiAliasFwd(torch::Tensor shading, torch::Tensor prim_id, torch::Tensor positions,
        torch::Tensor indices, const Viewport* viewport, const AntiAliasOppositeVertices* opposite_vertices)
    {
        AntiAliasOppositeVertices new_opposite_vertices;
        if (opposite_vertices == nullptr)
        {
            new_opposite_vertices = this->AntiAliasConstructOppositeVertices(indices);
            opposite_vertices = &new_opposite_vertices;
        }

        const uint32_t width = static_cast<uint32_t>(shading.size(2));
        const uint32_t height = static_cast<uint32_t>(shading.size(1));
        const uint32_t num_attribs = static_cast<uint32_t>(shading.size(3));

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(cmd_list, std::move(shading), aa_intermediate_.shading, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasFwd.shading");
        tensor_converter_.Convert(cmd_list, std::move(prim_id), aa_intermediate_.prim_id, GpuFormat::R32_Uint, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasFwd.prim_id");
        tensor_converter_.Convert(cmd_list, std::move(positions), aa_intermediate_.positions, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasFwd.positions");
        tensor_converter_.Convert(cmd_list, std::move(indices), aa_intermediate_.indices, GpuHeap::Default, GpuResourceFlag::None,
            L"GpuDiffRenderTorch.AntiAliasFwd.indices");

        if (viewport != nullptr)
        {
            aa_intermediate_.viewport.left = viewport->left;
            aa_intermediate_.viewport.top = viewport->top;
            aa_intermediate_.viewport.width = viewport->width;
            aa_intermediate_.viewport.height = viewport->height;
        }
        else
        {
            aa_intermediate_.viewport.left = 0;
            aa_intermediate_.viewport.top = 0;
            aa_intermediate_.viewport.width = static_cast<float>(width);
            aa_intermediate_.viewport.height = static_cast<float>(height);
        }
        aa_intermediate_.viewport.min_depth = 0;
        aa_intermediate_.viewport.max_depth = 1;

        gpu_dr_.AntiAliasFwd(cmd_list, aa_intermediate_.shading, aa_intermediate_.prim_id, aa_intermediate_.positions,
            aa_intermediate_.indices, aa_intermediate_.viewport, opposite_vertices->opposite_vertices, aa_intermediate_.anti_aliased);

        torch::Tensor anti_aliased =
            tensor_converter_.Convert(cmd_list, aa_intermediate_.anti_aliased, {1, height, width, num_attribs}, torch::kFloat32);

        gpu_system_.Execute(std::move(cmd_list));

        return anti_aliased;
    }

    std::tuple<torch::Tensor, torch::Tensor> GpuDiffRenderTorch::AntiAliasBwd(torch::Tensor grad_anti_aliased)
    {
        const uint32_t num_vertices = aa_intermediate_.positions.Size() / sizeof(glm::vec4);
        const uint32_t mini_batch = 1;
        const uint32_t width = aa_intermediate_.prim_id.Width(0);
        const uint32_t height = aa_intermediate_.prim_id.Height(0);
        const uint32_t num_attribs = aa_intermediate_.shading.Size() / (width * height * sizeof(float));

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(cmd_list, std::move(grad_anti_aliased), aa_intermediate_.grad_anti_aliased, GpuHeap::Default,
            GpuResourceFlag::None, L"GpuDiffRenderTorch.AntiAliasBwd.grad_anti_aliased");

        gpu_dr_.AntiAliasBwd(cmd_list, aa_intermediate_.shading, aa_intermediate_.prim_id, aa_intermediate_.positions,
            aa_intermediate_.indices, aa_intermediate_.viewport, aa_intermediate_.grad_anti_aliased, aa_intermediate_.grad_shading,
            aa_intermediate_.grad_positions);

        torch::Tensor grad_shading =
            tensor_converter_.Convert(cmd_list, aa_intermediate_.grad_shading, {mini_batch, height, width, num_attribs}, torch::kFloat32);
        torch::Tensor grad_positions =
            tensor_converter_.Convert(cmd_list, aa_intermediate_.grad_positions, {num_vertices, 4}, torch::kFloat32);

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(grad_shading), std::move(grad_positions)};
    }
} // namespace AIHoloImager
