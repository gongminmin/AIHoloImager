// Copyright (c) 2026 Minmin Gong
//

#include "GpuGSplatDiffRender/GpuGSplatDiffRender.hpp"

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"
#include "GpuComputationLib/Sorter/Sorter.hpp"

#include "CompiledShader/BlendFloat3Cs.h"
#include "CompiledShader/BlendUnorm4Cs.h"
#include "CompiledShader/PreprocessFwdCs.h"
#include "CompiledShader/RenderFwdGs.h"
#include "CompiledShader/RenderFwdPs.h"
#include "CompiledShader/RenderFwdVs.h"

namespace AIHoloImager
{
    class GpuGSplatDiffRender::Impl
    {
    public:
        explicit Impl(GpuSystem& gpu_system) : gpu_system_(gpu_system), sorter_(gpu_system)
        {
            {
                const ShaderInfo shader = {DEFINE_SHADER(PreprocessFwdCs)};
                preprocess_fwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            for (uint32_t i = 0; i < 2; ++i)
            {
                const ShaderInfo shaders[] = {
                    {DEFINE_SHADER(RenderFwdVs)},
                    {DEFINE_SHADER(RenderFwdPs)},
                    {DEFINE_SHADER(RenderFwdGs)},
                };

                const GpuFormat rtv_formats[] = {(i == 0) ? GSplatFmt : GpuFormat::RGBA32_Float};
                const GpuRenderPipeline::RenderTargetBlendDesc blend_desc{
                    .blend_enable = true,
                    .src_color_blend_factor = GpuRenderPipeline::BlendFactor::DstAlpha,
                    .dst_color_blend_factor = GpuRenderPipeline::BlendFactor::One,
                    .src_alpha_blend_factor = GpuRenderPipeline::BlendFactor::Zero,
                    .dst_alpha_blend_factor = GpuRenderPipeline::BlendFactor::InvSrcAlpha,
                };
                const GpuRenderPipeline::States states{
                    .cull_mode = GpuRenderPipeline::CullMode::CounterClockWise,
                    .rtv_formats = rtv_formats,
                    .blend_states{
                        .render_targets = std::span(&blend_desc, 1),
                    },
                };

                const GpuVertexLayout vertex_layout(gpu_system_, std::span<const GpuVertexAttrib>({
                                                                     {"TEXCOORD", 0, GpuFormat::RGBA32_Float, 0},
                                                                     {"TEXCOORD", 1, GpuFormat::RGB32_Float, 1},
                                                                     {"TEXCOORD", 2, GpuFormat::RGBA32_Float, 2},
                                                                 }));

                render_fwd_pipelines_[i] =
                    GpuRenderPipeline(gpu_system_, GpuRenderPipeline::PrimitiveTopology::PointList, shaders, vertex_layout, {}, states);
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(BlendUnorm4Cs)};
                blend_pipelines_[0] = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(BlendFloat3Cs)};
                blend_pipelines_[1] = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        void RenderFwd(GpuCommandList& cmd_list, const Gaussians& gaussians, const glm::mat4x4& model_mtx, const glm::mat4x4& view_mtx,
            const glm::mat4x4& proj_mtx, const GpuViewport& viewport, float kernel_size, GpuTexture2D* rendered_image,
            GpuBuffer* rendered_buff, uint32_t width, uint32_t height, const glm::vec3& bg_color)
        {
            assert((rendered_image != nullptr) != (rendered_buff != nullptr));

            const float tan_fov_x = 1 / proj_mtx[0][0];
            const float tan_fov_y = 1 / proj_mtx[1][1];
            const float focal_x = viewport.width * proj_mtx[0][0] / 2;
            const float focal_y = viewport.height * proj_mtx[1][1] / 2;

            GpuFormat gsplat_fmt;
            if (rendered_image != nullptr)
            {
                width = rendered_image->Width(0);
                height = rendered_image->Height(0);
                gsplat_fmt = GSplatFmt;
            }
            else
            {
                gsplat_fmt = GpuFormat::RGBA32_Float;
            }

            if (intermediate_cache_.num_gaussians_allocated < gaussians.num_gaussians)
            {
                intermediate_cache_.screen_pos_extents =
                    GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec4), GpuHeap::Default,
                        GpuResourceFlag::VertexBuffer | GpuResourceFlag::UnorderedAccess, "gsplat.intermediate_cache_.screen_pos");
                intermediate_cache_.conic_opacity = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec4), GpuHeap::Default,
                    GpuResourceFlag::VertexBuffer | GpuResourceFlag::UnorderedAccess, "gsplat.intermediate_cache_.conic_opacity");
                intermediate_cache_.color = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec3), GpuHeap::Default,
                    GpuResourceFlag::VertexBuffer | GpuResourceFlag::UnorderedAccess, "gsplat.intermediate_cache_.color");
                intermediate_cache_.num_visible_gaussians_indirect_args = GpuBuffer(gpu_system_, sizeof(GpuRenderIndexedArguments),
                    GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::IndirectArgs,
                    "gsplat.intermediate_cache_.num_visible_gaussians_indirect_args");
                intermediate_cache_.visible_key = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(uint32_t), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "gsplat.intermediate_cache_.visible_key");
                intermediate_cache_.visible_id = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(uint32_t), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::IndexBuffer | GpuResourceFlag::UnorderedAccess,
                    "gsplat.intermediate_cache_.visible_id");

                intermediate_cache_.num_gaussians_allocated = gaussians.num_gaussians;
            }
            if ((intermediate_cache_.gsplat_image.Width(0) != width) || (intermediate_cache_.gsplat_image.Height(0) != height))
            {
                intermediate_cache_.gsplat_image = GpuTexture2D(gpu_system_, width, height, 1, gsplat_fmt,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::RenderTarget, "gsplat.intermediate_cache_.gsplat_image");
            }

            {
                constexpr uint32_t BlockDim = 256;

                const glm::mat4x4 mv_mtx = view_mtx * model_mtx;
                const glm::vec4 eye_pos_os = glm::vec4(0, 0, 0, 1) * glm::inverse(mv_mtx);

                GpuConstantBufferOfType<PreprocessFwdConstantBuffer> preprocess_cb(gpu_system_, "preprocess_cb");
                preprocess_cb->num_gaussians = gaussians.num_gaussians;
                preprocess_cb->sh_degrees = gaussians.sh_degrees;
                preprocess_cb->num_coeffs = NumCoefficientsFromShDegrees(gaussians.sh_degrees);
                preprocess_cb->kernel_size = kernel_size;
                preprocess_cb->mv_mtx = glm::transpose(mv_mtx);
                preprocess_cb->mvp_mtx = glm::transpose(proj_mtx * mv_mtx);
                preprocess_cb->focal = {focal_x, focal_y};
                preprocess_cb->tan_fov = {tan_fov_x, tan_fov_y};
                preprocess_cb->vp_width_height = {viewport.width, viewport.height};
                preprocess_cb->eye_pos_os = glm::vec3(eye_pos_os.x, eye_pos_os.y, eye_pos_os.z) / eye_pos_os.w;
                preprocess_cb.UploadStaging();
                const GpuConstantBufferView preprocess_cbv(gpu_system_, preprocess_cb);

                const GpuShaderResourceView pos_srv(gpu_system_, gaussians.positions, GpuFormat::RGB32_Float);
                const GpuShaderResourceView scale_srv(gpu_system_, gaussians.scales, GpuFormat::RGB32_Float);
                const GpuShaderResourceView rotation_srv(gpu_system_, gaussians.rotations, GpuFormat::RGBA32_Float);
                const GpuShaderResourceView sh_srv(gpu_system_, gaussians.shs, GpuFormat::RGB32_Float);
                const GpuShaderResourceView opacity_srv(gpu_system_, gaussians.opacities, GpuFormat::R32_Float);

                GpuUnorderedAccessView screen_pos_extents_uav(gpu_system_, intermediate_cache_.screen_pos_extents, GpuFormat::RGBA32_Float);
                GpuUnorderedAccessView color_uav(gpu_system_, intermediate_cache_.color, GpuFormat::R32_Float);
                GpuUnorderedAccessView conic_opacity_uav(gpu_system_, intermediate_cache_.conic_opacity, GpuFormat::RGBA32_Float);
                GpuUnorderedAccessView num_visible_gaussians_uav(
                    gpu_system_, intermediate_cache_.num_visible_gaussians_indirect_args, GpuFormat::R32_Uint);
                GpuUnorderedAccessView visible_key_uav(gpu_system_, intermediate_cache_.visible_key, GpuFormat::R32_Uint);
                GpuUnorderedAccessView visible_id_uav(gpu_system_, intermediate_cache_.visible_id, GpuFormat::R32_Uint);

                {
                    const uint32_t clear_value[] = {0, 0, 0, 0};
                    cmd_list.Clear(num_visible_gaussians_uav, clear_value);
                }

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &preprocess_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"pos_buff", &pos_srv},
                    {"scale_buff", &scale_srv},
                    {"rotation_buff", &rotation_srv},
                    {"sh_buff", &sh_srv},
                    {"opacity_buff", &opacity_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"color_buff", &color_uav},
                    {"conic_opacity_buff", &conic_opacity_uav},
                    {"screen_pos_extents_buff", &screen_pos_extents_uav},
                    {"num_visible_gaussians_buff", &num_visible_gaussians_uav},
                    {"visible_key_buff", &visible_key_uav},
                    {"visible_id_buff", &visible_id_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(preprocess_fwd_pipeline_, {DivUp(gaussians.num_gaussians, BlockDim), 1, 1}, shader_binding);
            }

            {
                sorter_.RadixSort(cmd_list, intermediate_cache_.visible_key, GpuFormat::R32_Uint, intermediate_cache_.visible_id,
                    GpuFormat::R32_Uint, gaussians.num_gaussians, intermediate_cache_.visible_key, intermediate_cache_.visible_id, 32);
            }

            {
                GpuConstantBufferOfType<RenderFwdConstantBuffer> render_cb(gpu_system_, "render_cb");
                render_cb->viewport = {viewport.left, viewport.top, viewport.width, viewport.height};
                render_cb.UploadStaging();
                const GpuConstantBufferView render_cbv(gpu_system_, render_cb);

                GpuRenderTargetView gsplat_image_rtv(gpu_system_, intermediate_cache_.gsplat_image);

                {
                    const float init_value[] = {0, 0, 0, 1};
                    cmd_list.Clear(gsplat_image_rtv, init_value);
                }

                const GpuCommandList::VertexBufferBinding vb_bindings[] = {
                    {&intermediate_cache_.screen_pos_extents, 0},
                    {&intermediate_cache_.color, 0},
                    {&intermediate_cache_.conic_opacity, 0},
                };
                const GpuCommandList::IndexBufferBinding ib_binding = {&intermediate_cache_.visible_id, 0, GpuFormat::R32_Uint};

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &render_cbv},
                };
                const GpuCommandList::ShaderBinding shader_bindings[] = {
                    {cbvs, {}, {}},
                    {{}, {}, {}},
                    {{}, {}, {}},
                };

                GpuRenderTargetView* rtvs[] = {&gsplat_image_rtv};

                cmd_list.RenderIndexedIndirect(render_fwd_pipelines_[gsplat_fmt == GSplatFmt ? 0 : 1], vb_bindings, ib_binding,
                    intermediate_cache_.num_visible_gaussians_indirect_args, shader_bindings, rtvs, nullptr, std::span(&viewport, 1), {});
            }

            {
                constexpr uint32_t BlockDim = 16;

                GpuConstantBufferOfType<BlendConstantBuffer> blend_cb(gpu_system_, "blend_cb");
                blend_cb->width_height = {width, height};
                blend_cb->bg_color = bg_color;
                blend_cb->from_bg_color = (rendered_buff != nullptr);
                blend_cb.UploadStaging();
                const GpuConstantBufferView blend_cbv(gpu_system_, blend_cb);

                const GpuShaderResourceView gsplat_srv(
                    gpu_system_, intermediate_cache_.gsplat_image, 0, ToSRGBFormat(intermediate_cache_.gsplat_image.Format()));

                GpuUnorderedAccessView rendered_image_uav;
                if (rendered_image != nullptr)
                {
                    rendered_image_uav = GpuUnorderedAccessView(gpu_system_, *rendered_image, 0, ToLinearFormat(rendered_image->Format()));
                }
                else
                {
                    assert(rendered_buff != nullptr);
                    rendered_image_uav = GpuUnorderedAccessView(gpu_system_, *rendered_buff, GpuFormat::R32_Float);
                }

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &blend_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"gsplat_image", &gsplat_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"rendered_image", &rendered_image_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};

                cmd_list.Compute(
                    blend_pipelines_[rendered_buff != nullptr], {DivUp(width, BlockDim), DivUp(height, BlockDim), 1}, shader_binding);
            }
        }

        void RenderBwd([[maybe_unused]] GpuCommandList& cmd_list, [[maybe_unused]] const Gaussians& gaussians,
            [[maybe_unused]] const glm::mat4x4& model_mtx, [[maybe_unused]] const glm::mat4x4& view_mtx,
            [[maybe_unused]] const glm::mat4x4& proj_mtx, [[maybe_unused]] const GpuViewport& viewport, [[maybe_unused]] float kernel_size,
            [[maybe_unused]] const GpuBuffer& grad_image, [[maybe_unused]] GpuBuffer& grad_positions,
            [[maybe_unused]] GpuBuffer& grad_scales, [[maybe_unused]] GpuBuffer& grad_rotations, [[maybe_unused]] GpuBuffer& grad_dshs,
            [[maybe_unused]] GpuBuffer& grad_opacities)
        {
            // TODO
        }

    private:
        GpuSystem& gpu_system_;

        struct PreprocessFwdConstantBuffer
        {
            uint32_t num_gaussians;
            uint32_t sh_degrees;
            uint32_t num_coeffs;
            float kernel_size;

            glm::mat4x4 mv_mtx;
            glm::mat4x4 mvp_mtx;

            glm::vec2 focal;
            glm::vec2 tan_fov;

            glm::uvec2 vp_width_height;
            glm::uvec2 padding2;

            glm::vec3 eye_pos_os;
            float padding3;
        };
        GpuComputePipeline preprocess_fwd_pipeline_;

        struct RenderFwdConstantBuffer
        {
            glm::vec4 viewport;
        };
        GpuRenderPipeline render_fwd_pipelines_[2];

        struct BlendConstantBuffer
        {
            glm::uvec2 width_height;
            glm::uvec2 padding;
            glm::vec3 bg_color;
            uint32_t from_bg_color;
        };
        GpuComputePipeline blend_pipelines_[2];

        Sorter sorter_;

        struct IntermediateCache
        {
            uint32_t num_gaussians_allocated = 0;

            GpuBuffer screen_pos_extents;
            GpuBuffer conic_opacity;
            GpuBuffer color;
            GpuBuffer num_visible_gaussians_indirect_args;

            GpuBuffer visible_key;
            GpuBuffer visible_id;

            GpuTexture2D gsplat_image;
        };
        IntermediateCache intermediate_cache_;
        static constexpr GpuFormat GSplatFmt = GpuFormat::RGBA8_UNorm;
    };

    GpuGSplatDiffRender::GpuGSplatDiffRender() noexcept = default;
    GpuGSplatDiffRender::GpuGSplatDiffRender(GpuSystem& gpu_system) : impl_(std::make_unique<Impl>(gpu_system))
    {
    }

    GpuGSplatDiffRender::~GpuGSplatDiffRender() noexcept = default;

    GpuGSplatDiffRender::GpuGSplatDiffRender(GpuGSplatDiffRender&& other) noexcept = default;
    GpuGSplatDiffRender& GpuGSplatDiffRender::operator=(GpuGSplatDiffRender&& other) noexcept = default;

    void GpuGSplatDiffRender::RenderFwd(GpuCommandList& cmd_list, const Gaussians& gaussians, const glm::mat4x4& model_mtx,
        const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, const GpuViewport& viewport, float kernel_size,
        GpuTexture2D& rendered_image)
    {
        impl_->RenderFwd(
            cmd_list, gaussians, model_mtx, view_mtx, proj_mtx, viewport, kernel_size, &rendered_image, nullptr, 0, 0, glm::vec3(0, 0, 0));
    }

    void GpuGSplatDiffRender::RenderFwd(GpuCommandList& cmd_list, const Gaussians& gaussians, const glm::mat4x4& model_mtx,
        const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, const GpuViewport& viewport, float kernel_size, GpuBuffer& rendered_image,
        uint32_t width, uint32_t height, const glm::vec3& bg_color)
    {
        impl_->RenderFwd(
            cmd_list, gaussians, model_mtx, view_mtx, proj_mtx, viewport, kernel_size, nullptr, &rendered_image, width, height, bg_color);
    }

    void GpuGSplatDiffRender::RenderBwd(GpuCommandList& cmd_list, const Gaussians& gaussians, const glm::mat4x4& model_mtx,
        const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, const GpuViewport& viewport, float kernel_size,
        const GpuBuffer& grad_image, GpuBuffer& grad_positions, GpuBuffer& grad_scales, GpuBuffer& grad_rotations, GpuBuffer& grad_dshs,
        GpuBuffer& grad_opacities)
    {
        impl_->RenderBwd(cmd_list, gaussians, model_mtx, view_mtx, proj_mtx, viewport, kernel_size, grad_image, grad_positions, grad_scales,
            grad_rotations, grad_dshs, grad_opacities);
    }
} // namespace AIHoloImager
