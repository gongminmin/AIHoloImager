// Copyright (c) 2026 Minmin Gong
//

#include "GpuGSplatDiffRender/GpuGSplatDiffRender.hpp"

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"
#include "GpuComputationLib/Sorter/Sorter.hpp"

#include "CompiledShader/BlendFloat3Cs.h"
#include "CompiledShader/BlendUnorm4Cs.h"
#include "CompiledShader/CalcPixelsCs.h"
#include "CompiledShader/PpllGs.h"
#include "CompiledShader/PpllPs.h"
#include "CompiledShader/PpllVs.h"
#include "CompiledShader/PreprocessBwdCs.h"
#include "CompiledShader/PreprocessFwdCs.h"
#include "CompiledShader/PreprocessFwdCs_ForBwd.h"
#include "CompiledShader/RenderBwdCs.h"
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
                preprocess_fwd_pipelines_[0] = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(PreprocessFwdCs_ForBwd)};
                preprocess_fwd_pipelines_[1] = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(PreprocessBwdCs)};
                preprocess_bwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
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
            {
                const ShaderInfo shaders[] = {
                    {DEFINE_SHADER(PpllVs)},
                    {DEFINE_SHADER(PpllPs)},
                    {DEFINE_SHADER(PpllGs)},
                };

                const GpuRenderPipeline::States states{
                    .cull_mode = GpuRenderPipeline::CullMode::CounterClockWise,
                };

                const GpuVertexLayout vertex_layout(gpu_system, std::span<const GpuVertexAttrib>({
                                                                    {"TEXCOORD", 0, GpuFormat::RGBA32_Float, 0},
                                                                }));

                ppll_pipeline_ =
                    GpuRenderPipeline(gpu_system_, GpuRenderPipeline::PrimitiveTopology::PointList, shaders, vertex_layout, {}, states);
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(CalcPixelsCs)};
                calc_pixels_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(RenderBwdCs)};
                render_bwd_cs_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        void RenderFwd(GpuCommandList& cmd_list, const Gaussians& gaussians, const glm::mat4x4& model_mtx, const glm::mat4x4& view_mtx,
            const glm::mat4x4& proj_mtx, const GpuViewport& viewport, float kernel_size, GpuTexture2D* rendered_image,
            GpuBuffer* rendered_buff, uint32_t width, uint32_t height, const glm::vec3& bg_color, bool need_backward)
        {
            assert((rendered_image != nullptr) != (rendered_buff != nullptr));
            if (need_backward)
            {
                assert(rendered_buff != nullptr);
            }

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
                const GpuResourceFlag extra_flags = need_backward ? GpuResourceFlag::ShaderResource : GpuResourceFlag::None;

                intermediate_cache_.screen_pos_extents =
                    GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec4), GpuHeap::Default,
                        GpuResourceFlag::VertexBuffer | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured | extra_flags,
                        "gsplat.intermediate_cache_.screen_pos");
                intermediate_cache_.conic_opacity = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec4), GpuHeap::Default,
                    GpuResourceFlag::VertexBuffer | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured | extra_flags,
                    "gsplat.intermediate_cache_.conic_opacity");
                intermediate_cache_.color = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec3), GpuHeap::Default,
                    GpuResourceFlag::VertexBuffer | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured | extra_flags,
                    "gsplat.intermediate_cache_.color");
                intermediate_cache_.num_visible_gaussians_indirect_args =
                    GpuBuffer(gpu_system_, sizeof(GpuRenderIndexedArguments), GpuHeap::Default,
                        GpuResourceFlag::UnorderedAccess | GpuResourceFlag::IndirectArgs | GpuResourceFlag::Structured | extra_flags,
                        "gsplat.intermediate_cache_.num_visible_gaussians_indirect_args");
                intermediate_cache_.visible_key = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(uint32_t), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured,
                    "gsplat.intermediate_cache_.visible_key");
                intermediate_cache_.visible_id = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(uint32_t), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::IndexBuffer | GpuResourceFlag::UnorderedAccess |
                        GpuResourceFlag::Structured,
                    "gsplat.intermediate_cache_.visible_id");

                intermediate_cache_.num_gaussians_allocated = gaussians.num_gaussians;
            }
            if ((intermediate_cache_.gsplat_image.Width(0) != width) || (intermediate_cache_.gsplat_image.Height(0) != height) ||
                (intermediate_cache_.gsplat_image.Format() != gsplat_fmt))
            {
                intermediate_cache_.gsplat_image = GpuTexture2D(gpu_system_, width, height, 1, gsplat_fmt,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::RenderTarget, "gsplat.intermediate_cache_.gsplat_image");
            }

            if (need_backward)
            {
                intermediate_cache_.cov_3d = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(float) * 6, GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured,
                    "gsplat.intermediate_cache_.cov_3d");
                intermediate_cache_.clamped = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(uint32_t), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured,
                    "gsplat.intermediate_cache_.clamped");
            }
            else
            {
                intermediate_cache_.cov_3d = GpuBuffer();
                intermediate_cache_.clamped = GpuBuffer();
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

                const GpuShaderResourceView pos_srv(gpu_system_, gaussians.positions, sizeof(glm::vec3));
                const GpuShaderResourceView scale_srv(gpu_system_, gaussians.scales, sizeof(glm::vec3));
                const GpuShaderResourceView rotation_srv(gpu_system_, gaussians.rotations, sizeof(glm::vec4));
                const GpuShaderResourceView sh_srv(gpu_system_, gaussians.shs, sizeof(glm::vec3));
                const GpuShaderResourceView opacity_srv(gpu_system_, gaussians.opacities, sizeof(float));

                GpuUnorderedAccessView screen_pos_extents_uav(gpu_system_, intermediate_cache_.screen_pos_extents, sizeof(glm::vec4));
                GpuUnorderedAccessView color_uav(gpu_system_, intermediate_cache_.color, sizeof(float));
                GpuUnorderedAccessView conic_opacity_uav(gpu_system_, intermediate_cache_.conic_opacity, sizeof(glm::vec4));
                GpuUnorderedAccessView num_visible_gaussians_uav(
                    gpu_system_, intermediate_cache_.num_visible_gaussians_indirect_args, sizeof(uint32_t));
                GpuUnorderedAccessView visible_key_uav(gpu_system_, intermediate_cache_.visible_key, sizeof(uint32_t));
                GpuUnorderedAccessView visible_id_uav(gpu_system_, intermediate_cache_.visible_id, sizeof(uint32_t));

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
                std::vector<std::tuple<std::string_view, GpuUnorderedAccessView*>> uavs = {
                    {"color_buff", &color_uav},
                    {"conic_opacity_buff", &conic_opacity_uav},
                    {"screen_pos_extents_buff", &screen_pos_extents_uav},
                    {"num_visible_gaussians_buff", &num_visible_gaussians_uav},
                    {"visible_key_buff", &visible_key_uav},
                    {"visible_id_buff", &visible_id_uav},
                };
                GpuUnorderedAccessView cov_3d_uav;
                GpuUnorderedAccessView clamped_uav;
                if (need_backward)
                {
                    cov_3d_uav = GpuUnorderedAccessView(gpu_system_, intermediate_cache_.cov_3d, sizeof(float));
                    clamped_uav = GpuUnorderedAccessView(gpu_system_, intermediate_cache_.clamped, sizeof(uint32_t));

                    {
                        const float clear_value[] = {0, 0, 0, 0};
                        cmd_list.Clear(cov_3d_uav, clear_value);
                    }
                    {
                        const uint32_t clear_value[] = {0, 0, 0, 0};
                        cmd_list.Clear(clamped_uav, clear_value);
                    }

                    uavs.insert(uavs.end(), {
                                                {"cov_3d_buff", &cov_3d_uav},
                                                {"clamped_buff", &clamped_uav},
                                            });
                }

                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(
                    preprocess_fwd_pipelines_[need_backward], {DivUp(gaussians.num_gaussians, BlockDim), 1, 1}, shader_binding);
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
                    rendered_image_uav = GpuUnorderedAccessView(gpu_system_, *rendered_buff, sizeof(float));
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

        void RenderBwd(GpuCommandList& cmd_list, const Gaussians& gaussians, const glm::mat4x4& model_mtx, const glm::mat4x4& view_mtx,
            const glm::mat4x4& proj_mtx, const GpuViewport& viewport, float kernel_size, const GpuBuffer& grad_image,
            GpuBuffer& grad_positions, GpuBuffer& grad_scales, GpuBuffer& grad_rotations, GpuBuffer& grad_shs, GpuBuffer& grad_opacities)
        {
            const uint32_t width = intermediate_cache_.gsplat_image.Width(0);
            const uint32_t height = intermediate_cache_.gsplat_image.Height(0);

            if (grad_positions.Size() < gaussians.num_gaussians * sizeof(glm::vec3))
            {
                grad_positions = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec3), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable |
                        GpuResourceFlag::Structured,
                    "gsplat.grad_positions");
                grad_scales = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec3), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable |
                        GpuResourceFlag::Structured,
                    "gsplat.grad_scales");
                grad_rotations = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec4), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable |
                        GpuResourceFlag::Structured,
                    "gsplat.grad_rotations");
                grad_shs = GpuBuffer(gpu_system_,
                    gaussians.num_gaussians * NumCoefficientsFromShDegrees(gaussians.sh_degrees) * sizeof(glm::vec3), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable |
                        GpuResourceFlag::Structured,
                    "gsplat.grad_shs");
                grad_opacities = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(float), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable |
                        GpuResourceFlag::Structured,
                    "gsplat.grad_opacities");

                intermediate_cache_.grad_color = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec3), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured,
                    "gsplat.intermediate_cache_.grad_color");
                intermediate_cache_.grad_screen_pos = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec2), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured,
                    "gsplat.intermediate_cache_.grad_screen_pos");
                intermediate_cache_.grad_conic_2d = GpuBuffer(gpu_system_, gaussians.num_gaussians * sizeof(glm::vec3), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured,
                    "gsplat.intermediate_cache_.grad_conic_2d");
            }

            if (intermediate_cache_.pixels_counter.Size() < sizeof(uint32_t))
            {
                intermediate_cache_.pixels_counter = GpuBuffer(gpu_system_, sizeof(uint32_t), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured,
                    "gsplat.intermediate_cache_.pixels_counter");
            }
            {
                constexpr uint32_t BlockDim = 256;

                GpuConstantBufferOfType<CalcPixelConstantBuffer> calc_pixel_cb(gpu_system_, "calc_pixel_cb");
                calc_pixel_cb->viewport = {viewport.left, viewport.top, viewport.width, viewport.height};
                calc_pixel_cb->num_gaussians = gaussians.num_gaussians;
                calc_pixel_cb.UploadStaging();
                const GpuConstantBufferView calc_pixel_cbv(gpu_system_, calc_pixel_cb);

                const GpuShaderResourceView screen_pos_extents_srv(gpu_system_, intermediate_cache_.screen_pos_extents, sizeof(glm::vec4));

                GpuUnorderedAccessView pixels_counter_uav(gpu_system_, intermediate_cache_.pixels_counter, sizeof(uint32_t));

                {
                    const uint32_t init_value[] = {0, 0, 0, 0};
                    cmd_list.Clear(pixels_counter_uav, init_value);
                }

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &calc_pixel_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"screen_pos_extents_buff", &screen_pos_extents_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"pixels_counter_buff", &pixels_counter_uav},
                };

                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(calc_pixels_pipeline_, {DivUp(gaussians.num_gaussians, BlockDim), 1, 1}, shader_binding);
            }

            uint32_t num_pixels;
            {
                auto rb_future = cmd_list.ReadBackAsync(intermediate_cache_.pixels_counter, &num_pixels, sizeof(num_pixels));
                gpu_system_.ExecuteAndReset(cmd_list);
                rb_future.wait();
            }

            if ((intermediate_cache_.ppll_head.Width(0) != width) || (intermediate_cache_.ppll_head.Height(0) != height))
            {
                intermediate_cache_.ppll_head = GpuTexture2D(gpu_system_, width, height, 1, GpuFormat::R32_Uint,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "gsplat.ppll_head");
            }
            if (intermediate_cache_.ppll.Size() < num_pixels * sizeof(glm::uvec2))
            {
                intermediate_cache_.ppll = GpuBuffer(gpu_system_, num_pixels * sizeof(glm::uvec2), GpuHeap::Default,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Structured, "gsplat.ppll");
            }
            {
                GpuConstantBufferOfType<PpllVsConstantBuffer> ppll_vs_cb(gpu_system_, "ppll_vs_cb");
                ppll_vs_cb->viewport = {viewport.left, viewport.top, viewport.width, viewport.height};
                ppll_vs_cb.UploadStaging();
                const GpuConstantBufferView ppll_vs_cbv(gpu_system_, ppll_vs_cb);

                GpuUnorderedAccessView ppll_head_uav(gpu_system_, intermediate_cache_.ppll_head);
                GpuUnorderedAccessView ppll_uav(gpu_system_, intermediate_cache_.ppll, sizeof(uint32_t));
                GpuUnorderedAccessView ppll_counter_uav(gpu_system_, intermediate_cache_.pixels_counter, sizeof(uint32_t));

                {
                    const uint32_t init_value[] = {0, 0, 0, 0};
                    cmd_list.Clear(ppll_counter_uav, init_value);
                }
                {
                    const uint32_t init_value[] = {0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU};
                    cmd_list.Clear(ppll_head_uav, init_value);
                    cmd_list.Clear(ppll_uav, init_value);
                }

                const GpuCommandList::VertexBufferBinding vb_bindings[] = {
                    {&intermediate_cache_.screen_pos_extents, 0},
                };
                const GpuCommandList::IndexBufferBinding ib_binding = {&intermediate_cache_.visible_id, 0, GpuFormat::R32_Uint};

                std::tuple<std::string_view, const GpuConstantBufferView*> vs_cbvs[] = {
                    {"param_cb", &ppll_vs_cbv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> ps_uavs[] = {
                    {"ppll_head", &ppll_head_uav},
                    {"ppll", &ppll_uav},
                    {"ppll_counter", &ppll_counter_uav},
                };
                const GpuCommandList::ShaderBinding shader_bindings[] = {
                    {vs_cbvs, {}, {}},
                    {{}, {}, ps_uavs},
                    {{}, {}, {}},
                };

                cmd_list.RenderIndexedIndirect(ppll_pipeline_, vb_bindings, ib_binding,
                    intermediate_cache_.num_visible_gaussians_indirect_args, shader_bindings, {}, nullptr, std::span(&viewport, 1), {});
            }
            {
                constexpr uint32_t BlockDim = 16;

                GpuConstantBufferOfType<RenderBwdCsConstantBuffer> render_bwd_cb(gpu_system_, "render_bwd_cb");
                render_bwd_cb->viewport = {viewport.left, viewport.top, viewport.width, viewport.height};
                render_bwd_cb->bg_color = {0, 0, 0};
                render_bwd_cb->rt_width_height = {width, height};
                render_bwd_cb.UploadStaging();
                const GpuConstantBufferView render_bwd_cbv(gpu_system_, render_bwd_cb);

                const GpuShaderResourceView screen_pos_extents_srv(gpu_system_, intermediate_cache_.screen_pos_extents, sizeof(glm::vec4));
                const GpuShaderResourceView conic_opacity_srv(gpu_system_, intermediate_cache_.conic_opacity, sizeof(glm::vec4));
                const GpuShaderResourceView color_srv(gpu_system_, intermediate_cache_.color, sizeof(glm::vec3));

                const GpuShaderResourceView grad_image_srv(gpu_system_, grad_image, sizeof(glm::vec3));
                const GpuShaderResourceView gsplat_image_srv(gpu_system_, intermediate_cache_.gsplat_image);

                const GpuShaderResourceView ppll_head_srv(gpu_system_, intermediate_cache_.ppll_head);
                const GpuShaderResourceView ppll_srv(gpu_system_, intermediate_cache_.ppll, sizeof(glm::uvec2));

                GpuUnorderedAccessView grad_color_uav(gpu_system_, intermediate_cache_.grad_color, sizeof(float));
                GpuUnorderedAccessView grad_screen_pos_uav(gpu_system_, intermediate_cache_.grad_screen_pos, sizeof(float));
                GpuUnorderedAccessView grad_conic_2d_uav(gpu_system_, intermediate_cache_.grad_conic_2d, sizeof(float));
                GpuUnorderedAccessView grad_opacities_uav(gpu_system_, grad_opacities, sizeof(float));

                {
                    const uint32_t init_value[] = {0, 0, 0, 0};
                    cmd_list.Clear(grad_color_uav, init_value);
                    cmd_list.Clear(grad_screen_pos_uav, init_value);
                    cmd_list.Clear(grad_conic_2d_uav, init_value);
                    cmd_list.Clear(grad_opacities_uav, init_value);
                }

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &render_bwd_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"screen_pos_extents_buff", &screen_pos_extents_srv},
                    {"conic_opacity_buff", &conic_opacity_srv},
                    {"color_buff", &color_srv},
                    {"grad_image_buff", &grad_image_srv},
                    {"gsplat_image", &gsplat_image_srv},
                    {"ppll_head", &ppll_head_srv},
                    {"ppll", &ppll_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"grad_color_buff", &grad_color_uav},
                    {"grad_screen_pos_buff", &grad_screen_pos_uav},
                    {"grad_conic_2d_buff", &grad_conic_2d_uav},
                    {"grad_opacities_buff", &grad_opacities_uav},
                };

                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(render_bwd_cs_pipeline_, {DivUp(width, BlockDim), DivUp(height, BlockDim), 1}, shader_binding);
            }

            {
                constexpr uint32_t BlockDim = 256;

                const float tan_fov_x = 1 / proj_mtx[0][0];
                const float tan_fov_y = 1 / proj_mtx[1][1];
                const float focal_x = viewport.width * proj_mtx[0][0] / 2;
                const float focal_y = viewport.height * proj_mtx[1][1] / 2;

                const glm::mat4x4 mv_mtx = view_mtx * model_mtx;
                const glm::vec4 eye_pos_os = glm::vec4(0, 0, 0, 1) * glm::inverse(mv_mtx);

                GpuConstantBufferOfType<PreprocessBwdConstantBuffer> preprocess_cb(gpu_system_, "preprocess_cb");
                preprocess_cb->num_gaussians = gaussians.num_gaussians;
                preprocess_cb->sh_degrees = gaussians.sh_degrees;
                preprocess_cb->num_coeffs = NumCoefficientsFromShDegrees(gaussians.sh_degrees);
                preprocess_cb->kernel_size = kernel_size;
                preprocess_cb->mv_mtx = glm::transpose(mv_mtx);
                preprocess_cb->mvp_mtx = glm::transpose(proj_mtx * mv_mtx);
                preprocess_cb->focal = {focal_x, focal_y};
                preprocess_cb->tan_fov = {tan_fov_x, tan_fov_y};
                preprocess_cb->eye_pos_os = glm::vec3(eye_pos_os.x, eye_pos_os.y, eye_pos_os.z) / eye_pos_os.w;
                preprocess_cb.UploadStaging();
                const GpuConstantBufferView preprocess_cbv(gpu_system_, preprocess_cb);

                const GpuShaderResourceView pos_srv(gpu_system_, gaussians.positions, sizeof(glm::vec3));
                const GpuShaderResourceView screen_pos_extents_srv(gpu_system_, intermediate_cache_.screen_pos_extents, sizeof(glm::vec4));
                const GpuShaderResourceView cov_3d_srv(gpu_system_, intermediate_cache_.cov_3d, sizeof(float));
                const GpuShaderResourceView scale_srv(gpu_system_, gaussians.scales, sizeof(glm::vec3));
                const GpuShaderResourceView rotation_srv(gpu_system_, gaussians.rotations, sizeof(glm::vec4));
                const GpuShaderResourceView sh_srv(gpu_system_, gaussians.shs, sizeof(glm::vec3));
                const GpuShaderResourceView clamped_srv(gpu_system_, intermediate_cache_.clamped, sizeof(uint32_t));
                const GpuShaderResourceView conic_opacity_srv(gpu_system_, intermediate_cache_.conic_opacity, sizeof(glm::vec4));
                const GpuShaderResourceView grad_screen_pos_srv(gpu_system_, intermediate_cache_.grad_screen_pos, sizeof(glm::vec2));
                const GpuShaderResourceView grad_color_srv(gpu_system_, intermediate_cache_.grad_color, sizeof(glm::vec3));
                const GpuShaderResourceView grad_conic_2d_srv(gpu_system_, intermediate_cache_.grad_conic_2d, sizeof(glm::vec3));

                GpuUnorderedAccessView grad_positions_uav(gpu_system_, grad_positions, sizeof(float));
                GpuUnorderedAccessView grad_scales_uav(gpu_system_, grad_scales, sizeof(float));
                GpuUnorderedAccessView grad_rotations_uav(gpu_system_, grad_rotations, sizeof(glm::vec4));
                GpuUnorderedAccessView grad_shs_uav(gpu_system_, grad_shs, sizeof(float));
                GpuUnorderedAccessView grad_opacities_uav(gpu_system_, grad_opacities, sizeof(float));

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &preprocess_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"pos_buff", &pos_srv},
                    {"scale_buff", &scale_srv},
                    {"rotation_buff", &rotation_srv},
                    {"sh_buff", &sh_srv},
                    {"screen_pos_extents_buff", &screen_pos_extents_srv},
                    {"cov_3d_buff", &cov_3d_srv},
                    {"clamped_buff", &clamped_srv},
                    {"conic_opacity_buff", &conic_opacity_srv},
                    {"grad_screen_pos_buff", &grad_screen_pos_srv},
                    {"grad_color_buff", &grad_color_srv},
                    {"grad_conic_2d_buff", &grad_conic_2d_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"grad_positions_buff", &grad_positions_uav},
                    {"grad_scales_buff", &grad_scales_uav},
                    {"grad_rotations_buff", &grad_rotations_uav},
                    {"grad_shs_buff", &grad_shs_uav},
                    {"grad_opacities_buff", &grad_opacities_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(preprocess_bwd_pipeline_, {DivUp(gaussians.num_gaussians, BlockDim), 1, 1}, shader_binding);
            }
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
        GpuComputePipeline preprocess_fwd_pipelines_[2];

        struct PreprocessBwdConstantBuffer
        {
            uint32_t num_gaussians;
            uint32_t sh_degrees;
            uint32_t num_coeffs;
            float kernel_size;

            glm::mat4x4 mv_mtx;
            glm::mat4x4 mvp_mtx;

            glm::vec2 focal;
            glm::vec2 tan_fov;

            glm::vec3 eye_pos_os;
            uint32_t padding1;
        };
        GpuComputePipeline preprocess_bwd_pipeline_;

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

        struct CalcPixelConstantBuffer
        {
            glm::vec4 viewport;
            uint32_t num_gaussians;
            glm::uvec3 padding;
        };
        GpuComputePipeline calc_pixels_pipeline_;

        struct RenderBwdCsConstantBuffer
        {
            glm::vec4 viewport;
            glm::vec3 bg_color;
            float padding0;
            glm::uvec2 rt_width_height;
            glm::uvec2 padding1;
        };
        GpuComputePipeline render_bwd_cs_pipeline_;

        struct PpllVsConstantBuffer
        {
            glm::vec4 viewport;
        };
        GpuRenderPipeline ppll_pipeline_;

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

            // Only used for backward
            GpuBuffer cov_3d;
            GpuBuffer clamped;

            // Backward
            GpuBuffer grad_color;
            GpuBuffer grad_screen_pos;
            GpuBuffer grad_conic_2d;

            GpuBuffer pixels_counter;
            GpuTexture2D ppll_head;
            GpuBuffer ppll;
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
        impl_->RenderFwd(cmd_list, gaussians, model_mtx, view_mtx, proj_mtx, viewport, kernel_size, &rendered_image, nullptr, 0, 0,
            glm::vec3(0, 0, 0), false);
    }

    void GpuGSplatDiffRender::RenderFwd(GpuCommandList& cmd_list, const Gaussians& gaussians, const glm::mat4x4& model_mtx,
        const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, const GpuViewport& viewport, float kernel_size, GpuBuffer& rendered_image,
        uint32_t width, uint32_t height, const glm::vec3& bg_color, bool need_backward)
    {
        impl_->RenderFwd(cmd_list, gaussians, model_mtx, view_mtx, proj_mtx, viewport, kernel_size, nullptr, &rendered_image, width, height,
            bg_color, need_backward);
    }

    void GpuGSplatDiffRender::RenderBwd(GpuCommandList& cmd_list, const Gaussians& gaussians, const glm::mat4x4& model_mtx,
        const glm::mat4x4& view_mtx, const glm::mat4x4& proj_mtx, const GpuViewport& viewport, float kernel_size,
        const GpuBuffer& grad_image, GpuBuffer& grad_positions, GpuBuffer& grad_scales, GpuBuffer& grad_rotations, GpuBuffer& grad_shs,
        GpuBuffer& grad_opacities)
    {
        impl_->RenderBwd(cmd_list, gaussians, model_mtx, view_mtx, proj_mtx, viewport, kernel_size, grad_image, grad_positions, grad_scales,
            grad_rotations, grad_shs, grad_opacities);
    }
} // namespace AIHoloImager
