// Copyright (c) 2025 Minmin Gong
//

#include "GpuDiffRenderTorch.hpp"

#include <glm/vec4.hpp>

#include "Base/ErrorHandling.hpp"

using namespace torch::autograd;

namespace AIHoloImager
{
    GpuDiffRenderTorch::GpuDiffRenderTorch(size_t gpu_system, torch::Device torch_device)
        : gpu_system_(*reinterpret_cast<GpuSystem*>(gpu_system)), torch_device_(std::move(torch_device)), gpu_dr_(gpu_system_),
          tensor_converter_(gpu_system_, torch_device_)
    {
    }

    GpuDiffRenderTorch::~GpuDiffRenderTorch() = default;

    tensor_list GpuDiffRenderTorch::Rasterize(torch::Tensor positions, torch::Tensor indices, const std::array<uint32_t, 2>& resolution,
        const Viewport* viewport, bool needs_derivative_barycentric)
    {
        struct RasterizeAutogradFunc : public Function<RasterizeAutogradFunc>
        {
            static tensor_list forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor positions, torch::Tensor indices,
                const std::array<uint32_t, 2>& resolution, const Viewport* viewport, bool needs_derivative_barycentric)
            {
                auto [barycentric, prim_id, derivative_barycentric] =
                    dr->RasterizeFwd(std::move(positions), std::move(indices), resolution, viewport, needs_derivative_barycentric);
                ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                if (needs_derivative_barycentric)
                {
                    return {std::move(barycentric), std::move(prim_id), std::move(derivative_barycentric)};
                }
                else
                {
                    return {std::move(barycentric), std::move(prim_id)};
                }
            }

            static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
            {
                auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                torch::Tensor grad_barycentric = std::move(grad_outputs[0]);
                torch::Tensor grad_derivative_barycentric;
                if (grad_outputs.size() > 2)
                {
                    grad_derivative_barycentric = std::move(grad_outputs[2]);
                }
                auto grad_positions =
                    dr->RasterizeBwd(std::move(grad_barycentric), grad_outputs.size() > 2 ? &grad_derivative_barycentric : nullptr);
                return {torch::Tensor(), std::move(grad_positions), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
            }
        };

        return RasterizeAutogradFunc::apply(
            this, std::move(positions), std::move(indices), resolution, viewport, needs_derivative_barycentric);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GpuDiffRenderTorch::RasterizeFwd(torch::Tensor positions, torch::Tensor indices,
        const std::array<uint32_t, 2>& resolution, const Viewport* viewport, bool needs_derivative_barycentric)
    {
        const uint32_t width = resolution[0];
        const uint32_t height = resolution[1];

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(cmd_list, std::move(positions), rast_intermediate_.positions, GpuHeap::Default, GpuResourceFlag::None,
            "GpuDiffRenderTorch.RasterizeFwd.positions");
        tensor_converter_.Convert(cmd_list, std::move(indices), rast_intermediate_.indices, GpuHeap::Default, GpuResourceFlag::None,
            "GpuDiffRenderTorch.RasterizeFwd.indices");

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
            needs_derivative_barycentric, rast_intermediate_.barycentric, rast_intermediate_.derivative_barycentric,
            rast_intermediate_.prim_id);

        torch::Tensor barycentric = tensor_converter_.Convert(cmd_list, rast_intermediate_.barycentric);
        torch::Tensor prim_id = tensor_converter_.Convert(cmd_list, rast_intermediate_.prim_id);
        torch::Tensor derivative_barycentric;
        if (needs_derivative_barycentric)
        {
            derivative_barycentric = tensor_converter_.Convert(cmd_list, rast_intermediate_.derivative_barycentric);
        }

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(barycentric), std::move(prim_id), std::move(derivative_barycentric)};
    }

    torch::Tensor GpuDiffRenderTorch::RasterizeBwd(torch::Tensor grad_barycentric, torch::Tensor* grad_derivative_barycentric)
    {
        const uint32_t num_vertices = rast_intermediate_.positions.Size() / sizeof(glm::vec4);
        const bool needs_derivative_barycentric = grad_derivative_barycentric != nullptr;

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(cmd_list, std::move(grad_barycentric), rast_intermediate_.grad_barycentric, GpuFormat::RG32_Float,
            GpuResourceFlag::None, "GpuDiffRenderTorch.RasterizeBwd.grad_barycentric");
        if (needs_derivative_barycentric)
        {
            tensor_converter_.Convert(cmd_list, *grad_derivative_barycentric, rast_intermediate_.grad_derivative_barycentric,
                GpuFormat::RGBA32_Float, GpuResourceFlag::None, "GpuDiffRenderTorch.RasterizeBwd.grad_derivative_barycentric");
        }

        gpu_dr_.RasterizeBwd(cmd_list, rast_intermediate_.positions, rast_intermediate_.indices, rast_intermediate_.viewport,
            rast_intermediate_.barycentric, rast_intermediate_.prim_id, rast_intermediate_.grad_barycentric,
            rast_intermediate_.grad_derivative_barycentric, rast_intermediate_.grad_positions);

        const torch::Tensor grad_positions =
            tensor_converter_.Convert(cmd_list, rast_intermediate_.grad_positions, {num_vertices, 4}, torch::kFloat32);

        gpu_system_.Execute(std::move(cmd_list));

        return grad_positions;
    }

    tensor_list GpuDiffRenderTorch::Interpolate(torch::Tensor vtx_attribs, torch::Tensor barycentric, torch::Tensor prim_id,
        torch::Tensor indices, std::optional<torch::Tensor> derivative_barycentric)
    {
        if (derivative_barycentric.has_value())
        {
            struct InterpolateAutogradFunc : public Function<InterpolateAutogradFunc>
            {
                static tensor_list forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor vtx_attribs,
                    torch::Tensor barycentric, torch::Tensor derivative_barycentric, torch::Tensor prim_id, torch::Tensor indices)
                {
                    auto [shading, derivative_shading] = dr->InterpolateFwd(
                        std::move(vtx_attribs), std::move(barycentric), std::move(prim_id), std::move(indices), &derivative_barycentric);
                    ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                    return {std::move(shading), std::move(derivative_shading)};
                }

                static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
                {
                    auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                    torch::Tensor grad_shading = std::move(grad_outputs[0]);
                    torch::Tensor grad_derivative_shading = std::move(grad_outputs[1]);
                    auto [grad_vtx_attribs, grad_barycentric, grad_derivative_barycentric] =
                        dr->InterpolateBwd(std::move(grad_shading), &grad_derivative_shading);
                    return {torch::Tensor(), std::move(grad_vtx_attribs), std::move(grad_barycentric),
                        std::move(grad_derivative_barycentric), torch::Tensor(), torch::Tensor()};
                }
            };

            return InterpolateAutogradFunc::apply(
                this, std::move(vtx_attribs), std::move(barycentric), *derivative_barycentric, std::move(prim_id), std::move(indices));
        }
        else
        {
            struct InterpolateAutogradFunc : public Function<InterpolateAutogradFunc>
            {
                static tensor_list forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor vtx_attribs,
                    torch::Tensor barycentric, torch::Tensor prim_id, torch::Tensor indices)
                {
                    auto [shading, derivative_shading] =
                        dr->InterpolateFwd(std::move(vtx_attribs), std::move(barycentric), std::move(prim_id), std::move(indices), nullptr);
                    ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                    return {std::move(shading)};
                }

                static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
                {
                    auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                    torch::Tensor grad_shading = std::move(grad_outputs[0]);
                    auto [grad_vtx_attribs, grad_barycentric, grad_derivative_barycentric] =
                        dr->InterpolateBwd(std::move(grad_shading), nullptr);
                    return {torch::Tensor(), std::move(grad_vtx_attribs), std::move(grad_barycentric), torch::Tensor(), torch::Tensor()};
                }
            };

            return InterpolateAutogradFunc::apply(
                this, std::move(vtx_attribs), std::move(barycentric), std::move(prim_id), std::move(indices));
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> GpuDiffRenderTorch::InterpolateFwd(torch::Tensor vtx_attribs, torch::Tensor barycentric,
        torch::Tensor prim_id, torch::Tensor indices, const torch::Tensor* derivative_barycentric)
    {
        const uint32_t mini_batch = static_cast<uint32_t>(barycentric.size(0));
        const uint32_t width = static_cast<uint32_t>(barycentric.size(2));
        const uint32_t height = static_cast<uint32_t>(barycentric.size(1));
        const uint32_t num_attribs = static_cast<uint32_t>(vtx_attribs.size(1));
        const bool needs_dbc = derivative_barycentric != nullptr;

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        interpolate_intermediate_.num_attribs = num_attribs;
        tensor_converter_.Convert(cmd_list, std::move(vtx_attribs), interpolate_intermediate_.vtx_attribs, GpuHeap::Default,
            GpuResourceFlag::None, "GpuDiffRenderTorch.InterpolateFwd.vtx_attribs");
        tensor_converter_.Convert(cmd_list, std::move(barycentric), interpolate_intermediate_.barycentric, GpuFormat::RG32_Float,
            GpuResourceFlag::None, "GpuDiffRenderTorch.InterpolateFwd.barycentric");
        tensor_converter_.Convert(cmd_list, std::move(prim_id), interpolate_intermediate_.prim_id, GpuFormat::R32_Uint,
            GpuResourceFlag::None, "GpuDiffRenderTorch.InterpolateFwd.prim_id");
        tensor_converter_.Convert(cmd_list, std::move(indices), interpolate_intermediate_.indices, GpuHeap::Default, GpuResourceFlag::None,
            "GpuDiffRenderTorch.InterpolateFwd.indices");
        if (needs_dbc)
        {
            tensor_converter_.Convert(cmd_list, *derivative_barycentric, interpolate_intermediate_.derivative_barycentric,
                GpuFormat::RGBA32_Float, GpuResourceFlag::None, "GpuDiffRenderTorch.InterpolateFwd.derivative_barycentric");
        }
        else
        {
            interpolate_intermediate_.derivative_barycentric = GpuTexture2D();
        }

        gpu_dr_.InterpolateFwd(cmd_list, interpolate_intermediate_.vtx_attribs, num_attribs, interpolate_intermediate_.barycentric,
            interpolate_intermediate_.derivative_barycentric, interpolate_intermediate_.prim_id, interpolate_intermediate_.indices,
            interpolate_intermediate_.shading, interpolate_intermediate_.derivative_shading);

        torch::Tensor shading = tensor_converter_.Convert(
            cmd_list, interpolate_intermediate_.shading, {mini_batch, height, width, num_attribs}, torch::kFloat32);
        torch::Tensor derivative_shading;
        if (needs_dbc)
        {
            derivative_shading = tensor_converter_.Convert(
                cmd_list, interpolate_intermediate_.derivative_shading, {mini_batch, height, width, num_attribs * 2}, torch::kFloat32);
        }

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(shading), std::move(derivative_shading)};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GpuDiffRenderTorch::InterpolateBwd(
        torch::Tensor grad_shading, torch::Tensor* grad_derivative_shading)
    {
        const uint32_t num_attribs = interpolate_intermediate_.num_attribs;
        const uint32_t num_vertices = interpolate_intermediate_.vtx_attribs.Size() / (num_attribs * sizeof(float));
        const bool needs_dbc = static_cast<bool>(interpolate_intermediate_.derivative_barycentric);

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(cmd_list, std::move(grad_shading), interpolate_intermediate_.grad_shading, GpuHeap::Default,
            GpuResourceFlag::None, "GpuDiffRenderTorch.InterpolateBwd.grad_shading");
        if (needs_dbc)
        {
            tensor_converter_.Convert(cmd_list, *grad_derivative_shading, interpolate_intermediate_.grad_derivative_shading,
                GpuHeap::Default, GpuResourceFlag::None, "GpuDiffRenderTorch.InterpolateBwd.grad_derivative_shading");
        }
        else
        {
            interpolate_intermediate_.grad_derivative_shading = GpuBuffer();
        }

        gpu_dr_.InterpolateBwd(cmd_list, interpolate_intermediate_.vtx_attribs, num_attribs, interpolate_intermediate_.barycentric,
            interpolate_intermediate_.derivative_barycentric, interpolate_intermediate_.prim_id, interpolate_intermediate_.indices,
            interpolate_intermediate_.grad_shading, interpolate_intermediate_.grad_derivative_shading,
            interpolate_intermediate_.grad_vtx_attribs, interpolate_intermediate_.grad_barycentric,
            interpolate_intermediate_.grad_derivative_barycentric);

        torch::Tensor grad_vtx_attribs =
            tensor_converter_.Convert(cmd_list, interpolate_intermediate_.grad_vtx_attribs, {num_vertices, num_attribs}, torch::kFloat32);
        torch::Tensor grad_barycentric = tensor_converter_.Convert(cmd_list, interpolate_intermediate_.grad_barycentric);
        torch::Tensor grad_derivative_barycentric;
        if (needs_dbc)
        {
            grad_derivative_barycentric = tensor_converter_.Convert(cmd_list, interpolate_intermediate_.grad_derivative_barycentric);
        }

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(grad_vtx_attribs), std::move(grad_barycentric), std::move(grad_derivative_barycentric)};
    }

    torch::Tensor GpuDiffRenderTorch::Texture(torch::Tensor texture, torch::Tensor prim_id, torch::Tensor uv, std::string_view filter,
        std::string_view address_mode, std::optional<torch::Tensor> derivative_uv)
    {
        if (derivative_uv.has_value())
        {
            struct TextureAutogradFunc : public Function<TextureAutogradFunc>
            {
                static torch::Tensor forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor texture, torch::Tensor prim_id,
                    torch::Tensor uv, torch::Tensor derivative_uv, std::string_view filter, std::string_view address_mode)
                {
                    auto textured = dr->TextureFwd(
                        texture, std::move(prim_id), std::move(uv), std::move(filter), std::move(address_mode), &derivative_uv);
                    ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                    return textured;
                }

                static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
                {
                    auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                    torch::Tensor grad_textured = std::move(grad_outputs[0]);
                    auto [grad_texture, grad_uv, grad_derivative_uv] = dr->TextureBwd(std::move(grad_textured));
                    return {torch::Tensor(), std::move(grad_texture), torch::Tensor(), std::move(grad_uv), std::move(grad_derivative_uv),
                        torch::Tensor(), torch::Tensor(), torch::Tensor()};
                }
            };

            return TextureAutogradFunc::apply(
                this, std::move(texture), std::move(prim_id), std::move(uv), *derivative_uv, std::move(filter), std::move(address_mode));
        }
        else
        {
            struct TextureAutogradFunc : public Function<TextureAutogradFunc>
            {
                static torch::Tensor forward(AutogradContext* ctx, GpuDiffRenderTorch* dr, torch::Tensor texture, torch::Tensor prim_id,
                    torch::Tensor uv, std::string_view filter, std::string_view address_mode)
                {
                    auto textured = dr->TextureFwd(
                        std::move(texture), std::move(prim_id), std::move(uv), std::move(filter), std::move(address_mode), nullptr);
                    ctx->saved_data["dr"] = reinterpret_cast<int64_t>(dr);
                    return textured;
                }

                static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
                {
                    auto* dr = reinterpret_cast<GpuDiffRenderTorch*>(ctx->saved_data["dr"].to<int64_t>());

                    torch::Tensor grad_textured = std::move(grad_outputs[0]);
                    auto [grad_texture, grad_uv, grad_derivative_uv] = dr->TextureBwd(std::move(grad_textured));
                    return {torch::Tensor(), std::move(grad_texture), torch::Tensor(), std::move(grad_uv), torch::Tensor(), torch::Tensor(),
                        torch::Tensor()};
                }
            };

            return TextureAutogradFunc::apply(
                this, std::move(texture), std::move(prim_id), std::move(uv), std::move(filter), std::move(address_mode));
        }
    }

    torch::Tensor GpuDiffRenderTorch::TextureFwd(torch::Tensor texture, torch::Tensor prim_id, torch::Tensor uv, std::string_view filter,
        std::string_view address_mode, torch::Tensor* derivative_uv)
    {
        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        const torch::ScalarType scalar_type = texture.dtype().toScalarType();
        const uint32_t num_channels = static_cast<uint32_t>(texture.sizes().back());
        GpuFormat fmt = GpuFormat::Unknown;
        switch (scalar_type)
        {
        case torch::kUInt8:
            switch (num_channels)
            {
            case 1:
                fmt = GpuFormat::R8_UNorm;
                break;
            case 2:
                fmt = GpuFormat::RG8_UNorm;
                break;
            case 4:
                fmt = GpuFormat::RGBA8_UNorm;
                break;

            default:
                break;
            }
            break;

        case torch::kInt32:
            switch (num_channels)
            {
            case 1:
                fmt = GpuFormat::R32_Uint;
                break;
            case 2:
                fmt = GpuFormat::RG32_Uint;
                break;
            case 4:
                fmt = GpuFormat::RGBA32_Uint;
                break;

            default:
                break;
            }
            break;

        case torch::kFloat32:
            switch (num_channels)
            {
            case 1:
                fmt = GpuFormat::R32_Float;
                break;
            case 2:
                fmt = GpuFormat::RG32_Float;
                break;
            case 4:
                fmt = GpuFormat::RGBA32_Float;
                break;

            default:
                break;
            }
            break;

        default:
            break;
        }

        if (fmt == GpuFormat::Unknown)
        {
            Unreachable(std::format("Unsupported texture format {} channels: {}", static_cast<uint32_t>(scalar_type), num_channels));
        }

        tensor_converter_.Convert(cmd_list, std::move(texture), texture_intermediate_.texture, fmt, GpuResourceFlag::None,
            "GpuDiffRenderTorch.TextureFwd.texture");
        tensor_converter_.Convert(cmd_list, std::move(prim_id), texture_intermediate_.prim_id, GpuFormat::R32_Uint, GpuResourceFlag::None,
            "GpuDiffRenderTorch.TextureFwd.prim_id");
        tensor_converter_.Convert(
            cmd_list, std::move(uv), texture_intermediate_.uv, GpuHeap::Default, GpuResourceFlag::None, "GpuDiffRenderTorch.TextureFwd.uv");
        if (derivative_uv != nullptr)
        {
            tensor_converter_.Convert(cmd_list, *derivative_uv, texture_intermediate_.derivative_uv, GpuHeap::Default,
                GpuResourceFlag::None, "GpuDiffRenderTorch.TextureFwd.derivative_uv");
        }
        else
        {
            texture_intermediate_.derivative_uv = GpuBuffer();
        }

        if (filter == "auto")
        {
            filter = "linear-mipmap-linear";
        }

        GpuSampler::Filter min_mag_filter;
        GpuSampler::Filter mip_filter;
        bool needs_mip;
        if (filter == "point")
        {
            min_mag_filter = GpuSampler::Filter::Point;
            mip_filter = GpuSampler::Filter::Point;
            needs_mip = false;
        }
        else if (filter == "linear")
        {
            min_mag_filter = GpuSampler::Filter::Linear;
            mip_filter = GpuSampler::Filter::Point;
            needs_mip = false;
        }
        else if (filter == "point-mipmap-point")
        {
            min_mag_filter = GpuSampler::Filter::Point;
            mip_filter = GpuSampler::Filter::Point;
            needs_mip = true;
        }
        else if (filter == "point-mipmap-linear")
        {
            min_mag_filter = GpuSampler::Filter::Point;
            mip_filter = GpuSampler::Filter::Linear;
            needs_mip = true;
        }
        else if (filter == "linear-mipmap-point")
        {
            min_mag_filter = GpuSampler::Filter::Linear;
            mip_filter = GpuSampler::Filter::Point;
            needs_mip = true;
        }
        else if (filter == "linear-mipmap-linear")
        {
            min_mag_filter = GpuSampler::Filter::Linear;
            mip_filter = GpuSampler::Filter::Linear;
            needs_mip = true;
        }
        else
        {
            Unreachable(std::format("Unsupported sampler filter: {}", filter));
            min_mag_filter = GpuSampler::Filter::Point;
            mip_filter = GpuSampler::Filter::Point;
            needs_mip = false;
        }

        if (derivative_uv == nullptr)
        {
            needs_mip = false;
        }

        GpuSampler::AddressMode uvw_address_mode;
        if (address_mode == "wrap")
        {
            uvw_address_mode = GpuSampler::AddressMode::Wrap;
        }
        else if (address_mode == "mirror")
        {
            uvw_address_mode = GpuSampler::AddressMode::Mirror;
        }
        else if (address_mode == "clamp")
        {
            uvw_address_mode = GpuSampler::AddressMode::Clamp;
        }
        else
        {
            Unreachable(std::format("Unsupported sampler address mode: {}", address_mode));
        }

        const uint32_t mip_levels = needs_mip ? 0 : 1;
        gpu_dr_.GenerateMipmaps(cmd_list, texture_intermediate_.texture, mip_levels);

        texture_intermediate_.sampler = GpuDynamicSampler(
            gpu_system_, GpuSampler::Filters(min_mag_filter, min_mag_filter, mip_filter), GpuSampler::AddressModes(uvw_address_mode));

        gpu_dr_.TextureFwd(cmd_list, texture_intermediate_.texture, texture_intermediate_.prim_id, texture_intermediate_.uv,
            texture_intermediate_.derivative_uv, texture_intermediate_.sampler, texture_intermediate_.image);

        torch::Tensor image = tensor_converter_.Convert(cmd_list, texture_intermediate_.image);

        gpu_system_.Execute(std::move(cmd_list));

        return image;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GpuDiffRenderTorch::TextureBwd(torch::Tensor grad_image)
    {
        const uint32_t gbuffer_width = texture_intermediate_.prim_id.Width(0);
        const uint32_t gbuffer_height = texture_intermediate_.prim_id.Height(0);
        const uint32_t tex_width = texture_intermediate_.texture.Width(0);
        const uint32_t tex_height = texture_intermediate_.texture.Height(0);
        const GpuFormat format = texture_intermediate_.texture.Format();
        const uint32_t num_channels = FormatChannels(format);

        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        tensor_converter_.Convert(cmd_list, std::move(grad_image), texture_intermediate_.grad_image, format, GpuResourceFlag::None,
            "GpuDiffRenderTorch.TextureBwd.grad_image");

        gpu_dr_.TextureBwd(cmd_list, texture_intermediate_.texture, texture_intermediate_.prim_id, texture_intermediate_.uv,
            texture_intermediate_.derivative_uv, texture_intermediate_.grad_image, texture_intermediate_.sampler,
            texture_intermediate_.grad_texture, texture_intermediate_.grad_uv, texture_intermediate_.grad_derivative_uv);

        torch::Tensor grad_texture = tensor_converter_.Convert(
            cmd_list, texture_intermediate_.grad_texture, {1, tex_height, tex_width, num_channels}, torch::kFloat32);
        torch::Tensor grad_uv =
            tensor_converter_.Convert(cmd_list, texture_intermediate_.grad_uv, {1, gbuffer_height, gbuffer_width, 2}, torch::kFloat32);
        torch::Tensor grad_derivative_uv;
        if (static_cast<bool>(texture_intermediate_.grad_derivative_uv))
        {
            grad_derivative_uv = tensor_converter_.Convert(
                cmd_list, texture_intermediate_.grad_derivative_uv, {1, gbuffer_height, gbuffer_width, 4}, torch::kFloat32);
        }

        gpu_system_.Execute(std::move(cmd_list));

        return {std::move(grad_texture), std::move(grad_uv), std::move(grad_derivative_uv)};
    }

    GpuDiffRenderTorch::AntiAliasOppositeVertices GpuDiffRenderTorch::AntiAliasConstructOppositeVertices(torch::Tensor indices)
    {
        auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

        GpuBuffer indices_buff;
        tensor_converter_.Convert(cmd_list, std::move(indices), indices_buff, GpuHeap::Default, GpuResourceFlag::None,
            "GpuDiffRenderTorch.AntiAliasConstructOppositeVertices.indices");

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
            "GpuDiffRenderTorch.AntiAliasFwd.shading");
        tensor_converter_.Convert(cmd_list, std::move(prim_id), aa_intermediate_.prim_id, GpuFormat::R32_Uint, GpuResourceFlag::None,
            "GpuDiffRenderTorch.AntiAliasFwd.prim_id");
        tensor_converter_.Convert(cmd_list, std::move(positions), aa_intermediate_.positions, GpuHeap::Default, GpuResourceFlag::None,
            "GpuDiffRenderTorch.AntiAliasFwd.positions");
        tensor_converter_.Convert(cmd_list, std::move(indices), aa_intermediate_.indices, GpuHeap::Default, GpuResourceFlag::None,
            "GpuDiffRenderTorch.AntiAliasFwd.indices");

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
            GpuResourceFlag::None, "GpuDiffRenderTorch.AntiAliasBwd.grad_anti_aliased");

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
