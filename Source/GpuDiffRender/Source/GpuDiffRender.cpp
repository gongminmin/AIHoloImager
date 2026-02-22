// Copyright (c) 2025-2026 Minmin Gong
//

#include "GpuDiffRender.hpp"

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Base/Util.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuConstantBuffer.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuTexture.hpp"

#include "CompiledShader/AccumGradMipsCs.h"
#include "CompiledShader/AntiAliasConstructOppoVertCs.h"
#include "CompiledShader/AntiAliasConstructOppoVertHashCs.h"
#include "CompiledShader/AntialiasBwdCs.h"
#include "CompiledShader/AntialiasFwdCs.h"
#include "CompiledShader/AntialiasIndirectCs.h"
#include "CompiledShader/InterpolateBwdCs.h"
#include "CompiledShader/InterpolateBwdDerivateAttribsCs.h"
#include "CompiledShader/InterpolateFwdCs.h"
#include "CompiledShader/InterpolateFwdDerivateAttribsCs.h"
#include "CompiledShader/RasterizeBwdCs.h"
#include "CompiledShader/RasterizeBwdDerivateBcCs.h"
#include "CompiledShader/RasterizeFwdDerivateBcGs.h"
#include "CompiledShader/RasterizeFwdDerivateBcPs.h"
#include "CompiledShader/RasterizeFwdGs.h"
#include "CompiledShader/RasterizeFwdPs.h"
#include "CompiledShader/RasterizeFwdVs.h"
#include "CompiledShader/TextureBwdCs.h"
#include "CompiledShader/TextureBwdMipCs.h"
#include "CompiledShader/TextureCopyCs.h"
#include "CompiledShader/TextureFwdCs.h"
#include "CompiledShader/TextureFwdMipCs.h"

namespace AIHoloImager
{
    GpuDiffRender::GpuDiffRender(GpuSystem& gpu_system) : gpu_system_(gpu_system)
    {
        {
            GpuRenderPipeline::States states;
            states.cull_mode = GpuRenderPipeline::CullMode::ClockWise;
            states.dsv_format = GpuFormat::D32_Float;
            states.depth_enable = true;

            const GpuVertexLayout vertex_layout(gpu_system_, std::span<const GpuVertexAttrib>({
                                                                 {"POSITION", 0, GpuFormat::RGBA32_Float},
                                                             }));

            {
                const ShaderInfo shaders[] = {
                    {DEFINE_SHADER(RasterizeFwdVs)},
                    {DEFINE_SHADER(RasterizeFwdPs)},
                    {DEFINE_SHADER(RasterizeFwdGs)},
                };

                const GpuFormat rtv_formats[] = {GpuFormat::RG32_Float, GpuFormat::R32_Uint};
                states.rtv_formats = rtv_formats;

                rasterize_fwd_pipeline_ =
                    GpuRenderPipeline(gpu_system_, GpuRenderPipeline::PrimitiveTopology::TriangleList, shaders, vertex_layout, {}, states);
            }
            {
                const ShaderInfo shaders[] = {
                    {DEFINE_SHADER(RasterizeFwdVs)},
                    {DEFINE_SHADER(RasterizeFwdDerivateBcPs)},
                    {DEFINE_SHADER(RasterizeFwdDerivateBcGs)},
                };

                const GpuFormat rtv_formats[] = {GpuFormat::RG32_Float, GpuFormat::R32_Uint, GpuFormat::RGBA32_Float};
                states.rtv_formats = rtv_formats;

                rasterize_fwd_derivative_bc_pipeline_ =
                    GpuRenderPipeline(gpu_system_, GpuRenderPipeline::PrimitiveTopology::TriangleList, shaders, vertex_layout, {}, states);
            }
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(RasterizeBwdCs)};
            rasterize_bwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(RasterizeBwdDerivateBcCs)};
            rasterize_bwd_derivative_bc_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }

        {
            const ShaderInfo shader = {DEFINE_SHADER(InterpolateFwdCs)};
            interpolate_fwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(InterpolateFwdDerivateAttribsCs)};
            interpolate_fwd_derivative_attribs_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(InterpolateBwdCs)};
            interpolate_bwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(InterpolateBwdDerivateAttribsCs)};
            interpolate_bwd_derivative_attribs_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }

        {
            const ShaderInfo shader = {DEFINE_SHADER(TextureCopyCs)};
            texture_copy_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(TextureFwdCs)};
            texture_fwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(TextureFwdMipCs)};
            texture_fwd_mip_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(TextureBwdCs)};
            texture_bwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(TextureBwdMipCs)};
            texture_bwd_mip_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(AccumGradMipsCs)};
            accum_grad_mips_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }

        {
            const ShaderInfo shader = {DEFINE_SHADER(AntiAliasConstructOppoVertHashCs)};
            anti_alias_construct_oppo_vert_hash_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(AntiAliasConstructOppoVertCs)};
            anti_alias_construct_oppo_vert_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            anti_alias_indirect_args_cb_ =
                GpuConstantBufferOfType<AntiAliasIndirectArgsConstantBuffer>(gpu_system_, "anti_alias_indirect_args_cb_");
            anti_alias_indirect_args_cb_->bwd_block_dim = 256;
            anti_alias_indirect_args_cb_.UploadStaging();

            anti_alias_indirect_args_cbv_ = GpuConstantBufferView(gpu_system_, anti_alias_indirect_args_cb_);

            const ShaderInfo shader = {DEFINE_SHADER(AntiAliasIndirectCs)};
            anti_alias_indirect_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(AntiAliasFwdCs)};
            anti_alias_fwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {DEFINE_SHADER(AntiAliasBwdCs)};
            anti_alias_bwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }

        silhouette_counter_ = GpuBuffer(gpu_system_, sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::UnorderedAccess,
            "GpuDiffRender.AntiAliasFwd.silhouette_counter");
        silhouette_counter_srv_ = GpuShaderResourceView(gpu_system_, silhouette_counter_, GpuFormat::R32_Uint);
        silhouette_counter_uav_ = GpuUnorderedAccessView(gpu_system_, silhouette_counter_, GpuFormat::R32_Uint);

        indirect_args_ = GpuBuffer(gpu_system_, 3 * sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::UnorderedAccess,
            "GpuDiffRender.AntiAliasBwd.indirect_args");
        indirect_args_uav_ = GpuUnorderedAccessView(gpu_system_, indirect_args_, GpuFormat::R32_Uint);
    }

    GpuDiffRender::~GpuDiffRender() = default;

    void GpuDiffRender::RasterizeFwd(GpuCommandList& cmd_list, const GpuBuffer& positions, const GpuBuffer& indices, uint32_t width,
        uint32_t height, const GpuViewport& viewport, bool needs_derivative_barycentric, GpuTexture2D& barycentric,
        GpuTexture2D& derivative_barycentric, GpuTexture2D& prim_id)
    {
        if ((barycentric.Width(0) != width) || (barycentric.Height(0) != height) || (barycentric.Format() != GpuFormat::RG32_Float))
        {
            barycentric = GpuTexture2D(
                gpu_system_, width, height, 1, GpuFormat::RG32_Float, GpuResourceFlag::RenderTarget | GpuResourceFlag::Shareable);
        }
        barycentric.Name("GpuDiffRender.RasterizeFwd.barycentric");

        if ((prim_id.Width(0) != width) || (prim_id.Height(0) != height) || (prim_id.Format() != GpuFormat::R32_Uint))
        {
            prim_id = GpuTexture2D(
                gpu_system_, width, height, 1, GpuFormat::R32_Uint, GpuResourceFlag::RenderTarget | GpuResourceFlag::Shareable);
        }
        prim_id.Name("GpuDiffRender.RasterizeFwd.prim_id");

        if (needs_derivative_barycentric)
        {
            if ((derivative_barycentric.Width(0) != width) || (derivative_barycentric.Height(0) != height) ||
                (derivative_barycentric.Format() != GpuFormat::RGBA32_Float))
            {
                derivative_barycentric = GpuTexture2D(
                    gpu_system_, width, height, 1, GpuFormat::RGBA32_Float, GpuResourceFlag::RenderTarget | GpuResourceFlag::Shareable);
            }
            derivative_barycentric.Name("GpuDiffRender.RasterizeFwd.derivative_barycentric");
        }
        else
        {
            derivative_barycentric = GpuTexture2D();
        }

        GpuRenderTargetView barycentric_rtv(gpu_system_, barycentric);
        GpuRenderTargetView prim_id_rtv(gpu_system_, prim_id);
        GpuRenderTargetView derivative_barycentric_rtv;
        if (needs_derivative_barycentric)
        {
            derivative_barycentric_rtv = GpuRenderTargetView(gpu_system_, derivative_barycentric);
        }

        if ((depth_tex_.Width(0) != width) || (depth_tex_.Height(0) != height))
        {
            depth_tex_ = GpuTexture2D(
                gpu_system_, width, height, 1, GpuFormat::D32_Float, GpuResourceFlag::DepthStencil, "GpuDiffRender.RasterizeFwd.depth_tex");
            depth_srv_ = GpuShaderResourceView(gpu_system_, depth_tex_, GpuFormat::R32_Float);
            depth_dsv_ = GpuDepthStencilView(gpu_system_, depth_tex_);
        }

        const float clear_clr[] = {0, 0, 0, 0};
        cmd_list.Clear(prim_id_rtv, clear_clr);
        cmd_list.ClearDepth(depth_dsv_, 1.0f);
        if (needs_derivative_barycentric)
        {
            cmd_list.Clear(derivative_barycentric_rtv, clear_clr);
        }

        const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&positions, 0}};
        const GpuCommandList::IndexBufferBinding ib_binding = {&indices, 0, GpuFormat::R32_Uint};

        if (needs_derivative_barycentric)
        {
            GpuConstantBufferOfType<RasterizeFwdPsDerivateBcConstantBuffer> rasterize_fwd_ps_derivative_bc_cb(
                gpu_system_, "rasterize_fwd_ps_derivative_bc_cb");
            rasterize_fwd_ps_derivative_bc_cb->viewport = glm::vec4(viewport.left, viewport.top, viewport.width, viewport.height);
            rasterize_fwd_ps_derivative_bc_cb.UploadStaging();

            const GpuConstantBufferView rasterize_fwd_ps_derivative_bc_cbv(gpu_system_, rasterize_fwd_ps_derivative_bc_cb);

            const GpuShaderResourceView positions_srv(gpu_system_, positions, GpuFormat::RGBA32_Float);
            const GpuShaderResourceView indices_srv(gpu_system_, indices, GpuFormat::R32_Uint);

            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                {"param_cb", &rasterize_fwd_ps_derivative_bc_cbv},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"positions_buff", &positions_srv},
                {"indices_buff", &indices_srv},
            };
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {{}, {}, {}},
                {cbvs, srvs, {}},
                {{}, {}, {}},
            };

            GpuRenderTargetView* rtvs[] = {&barycentric_rtv, &prim_id_rtv, &derivative_barycentric_rtv};

            cmd_list.Render(rasterize_fwd_derivative_bc_pipeline_, vb_bindings, &ib_binding, indices.Size() / sizeof(uint32_t),
                shader_bindings, rtvs, &depth_dsv_, std::span(&viewport, 1), {});
        }
        else
        {
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {{}, {}, {}},
                {{}, {}, {}},
                {{}, {}, {}},
            };

            GpuRenderTargetView* rtvs[] = {&barycentric_rtv, &prim_id_rtv};

            cmd_list.Render(rasterize_fwd_pipeline_, vb_bindings, &ib_binding, indices.Size() / sizeof(uint32_t), shader_bindings, rtvs,
                &depth_dsv_, std::span(&viewport, 1), {});
        }
    }

    void GpuDiffRender::RasterizeBwd(GpuCommandList& cmd_list, const GpuBuffer& positions, const GpuBuffer& indices,
        const GpuViewport& viewport, const GpuTexture2D& barycentric, const GpuTexture2D& prim_id, const GpuTexture2D& grad_barycentric,
        const GpuTexture2D& grad_derivative_barycentric, GpuBuffer& grad_positions)
    {
        const uint32_t width = barycentric.Width(0);
        const uint32_t height = barycentric.Height(0);
        const bool needs_derivative_barycentric = static_cast<bool>(grad_derivative_barycentric);

        if (grad_positions.Size() != positions.Size())
        {
            grad_positions =
                GpuBuffer(gpu_system_, positions.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_positions.Name("GpuDiffRender.RasterizeBwd.grad_positions");

        constexpr uint32_t BlockDim = 16;

        GpuConstantBufferOfType<RasterizeBwdConstantBuffer> rasterize_bwd_cb(gpu_system_, "rasterize_bwd_cb");
        rasterize_bwd_cb->viewport = glm::vec4(viewport.left, viewport.top, viewport.width, viewport.height);
        rasterize_bwd_cb->gbuffer_size = glm::uvec2(width, height);
        rasterize_bwd_cb.UploadStaging();

        const GpuConstantBufferView rasterize_bwd_cbv(gpu_system_, rasterize_bwd_cb);

        const GpuShaderResourceView barycentric_srv(gpu_system_, barycentric, GpuFormat::RG32_Float);
        const GpuShaderResourceView prim_id_srv(gpu_system_, prim_id, GpuFormat::R32_Uint);
        const GpuShaderResourceView grad_barycentric_srv(gpu_system_, grad_barycentric, GpuFormat::RG32_Float);
        const GpuShaderResourceView positions_srv(gpu_system_, positions, GpuFormat::RGBA32_Float);
        const GpuShaderResourceView indices_srv(gpu_system_, indices, GpuFormat::R32_Uint);
        GpuShaderResourceView grad_derivative_barycentric_srv;
        if (needs_derivative_barycentric)
        {
            grad_derivative_barycentric_srv = GpuShaderResourceView(gpu_system_, grad_derivative_barycentric, GpuFormat::RGBA32_Float);
        }

        GpuUnorderedAccessView grad_positions_uav(
            gpu_system_, grad_positions, GpuFormat::R32_Uint); // Float as uint due to atomic operations

        const uint32_t clear_clr[] = {0, 0, 0, 0};
        cmd_list.Clear(grad_positions_uav, clear_clr);

        std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
            {"param_cb", &rasterize_bwd_cbv},
        };
        std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
            {"barycentric_tex", &barycentric_srv},
            {"prim_id_tex", &prim_id_srv},
            {"grad_barycentric_tex", &grad_barycentric_srv},
            {"positions_buff", &positions_srv},
            {"indices_buff", &indices_srv},
            {"grad_derivative_barycentric_tex", &grad_derivative_barycentric_srv},
        };
        std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
            {"grad_positions_buff", &grad_positions_uav},
        };
        const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
        cmd_list.Compute(needs_derivative_barycentric ? rasterize_bwd_derivative_bc_pipeline_ : rasterize_bwd_pipeline_,
            DivUp(width, BlockDim), DivUp(height, BlockDim), 1, shader_binding);
    }

    void GpuDiffRender::InterpolateFwd(GpuCommandList& cmd_list, const GpuBuffer& vtx_attribs, uint32_t num_attribs_per_vtx,
        const GpuTexture2D& barycentric, const GpuTexture2D& derivative_barycentric, const GpuTexture2D& prim_id, const GpuBuffer& indices,
        GpuBuffer& shading, GpuBuffer& derivative_shading)
    {
        const uint32_t width = barycentric.Width(0);
        const uint32_t height = barycentric.Height(0);
        const bool needs_dbc = static_cast<bool>(derivative_barycentric);

        const uint32_t shading_size = width * height * num_attribs_per_vtx * sizeof(float);
        if (shading.Size() != shading_size)
        {
            shading = GpuBuffer(gpu_system_, shading_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        shading.Name("GpuDiffRender.InterpolateFwd.shading");

        if (needs_dbc)
        {
            const uint32_t derivative_shading_size = shading_size * 2;
            if (derivative_shading.Size() != derivative_shading_size)
            {
                derivative_shading = GpuBuffer(
                    gpu_system_, derivative_shading_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
            }
            derivative_shading.Name("GpuDiffRender.InterpolateFwd.derivative_shading");
        }
        else
        {
            derivative_shading = GpuBuffer();
        }

        constexpr uint32_t BlockDim = 16;

        GpuConstantBufferOfType<InterpolateFwdConstantBuffer> interpolate_fwd_cb(gpu_system_, "interpolate_fwd_cb");
        interpolate_fwd_cb->gbuffer_size = glm::uvec2(width, height);
        interpolate_fwd_cb->num_attribs = num_attribs_per_vtx;
        interpolate_fwd_cb.UploadStaging();

        const GpuConstantBufferView interpolate_fwd_cbv(gpu_system_, interpolate_fwd_cb);

        const GpuShaderResourceView barycentric_srv(gpu_system_, barycentric, GpuFormat::RG32_Float);
        const GpuShaderResourceView prim_id_srv(gpu_system_, prim_id, GpuFormat::R32_Uint);
        const GpuShaderResourceView attrib_srv(gpu_system_, vtx_attribs, GpuFormat::R32_Float);
        const GpuShaderResourceView index_srv(gpu_system_, indices, GpuFormat::R32_Uint);
        GpuShaderResourceView derivative_barycentric_srv;
        if (needs_dbc)
        {
            derivative_barycentric_srv = GpuShaderResourceView(gpu_system_, derivative_barycentric, GpuFormat::RGBA32_Float);
        }

        GpuUnorderedAccessView shading_uav(gpu_system_, shading, GpuFormat::R32_Float);
        GpuUnorderedAccessView derivative_shading_uav;
        if (needs_dbc)
        {
            derivative_shading_uav = GpuUnorderedAccessView(gpu_system_, derivative_shading, GpuFormat::RG32_Float);
        }

        const float clear_clr[] = {0, 0, 0, 0};
        cmd_list.Clear(shading_uav, clear_clr);
        if (needs_dbc)
        {
            cmd_list.Clear(derivative_shading_uav, clear_clr);
        }

        std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
            {"param_cb", &interpolate_fwd_cbv},
        };
        std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
            {"barycentric_tex", &barycentric_srv},
            {"prim_id_tex", &prim_id_srv},
            {"vtx_attribs_buff", &attrib_srv},
            {"indices_buff", &index_srv},
            {"derivative_barycentric_tex", &derivative_barycentric_srv},
        };
        std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
            {"shading", &shading_uav},
            {"derivative_shading", &derivative_shading_uav},
        };
        const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
        cmd_list.Compute(needs_dbc ? interpolate_fwd_derivative_attribs_pipeline_ : interpolate_fwd_pipeline_, DivUp(width, BlockDim),
            DivUp(height, BlockDim), 1, shader_binding);
    }

    void GpuDiffRender::InterpolateBwd(GpuCommandList& cmd_list, const GpuBuffer& vtx_attribs, uint32_t num_attribs_per_vtx,
        const GpuTexture2D& barycentric, const GpuTexture2D& derivative_barycentric, const GpuTexture2D& prim_id, const GpuBuffer& indices,
        const GpuBuffer& grad_shading, const GpuBuffer& grad_derivative_shading, GpuBuffer& grad_vtx_attribs,
        GpuTexture2D& grad_barycentric, GpuTexture2D& grad_derivative_barycentric)
    {
        const uint32_t width = barycentric.Width(0);
        const uint32_t height = barycentric.Height(0);
        const bool needs_dbc = static_cast<bool>(derivative_barycentric);

        if (grad_vtx_attribs.Size() != vtx_attribs.Size())
        {
            grad_vtx_attribs =
                GpuBuffer(gpu_system_, vtx_attribs.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_vtx_attribs.Name("GpuDiffRender.InterpolateBwd.grad_vtx_attribs");

        if ((grad_barycentric.Width(0) != width) || (grad_barycentric.Height(0) != height) ||
            (grad_barycentric.Format() != GpuFormat::RG32_Float))
        {
            grad_barycentric = GpuTexture2D(
                gpu_system_, width, height, 1, GpuFormat::RG32_Float, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_barycentric.Name("GpuDiffRender.InterpolateBwd.grad_barycentric");

        if (needs_dbc)
        {
            if ((grad_derivative_barycentric.Width(0) != width) || (grad_derivative_barycentric.Height(0) != height) ||
                (grad_derivative_barycentric.Format() != GpuFormat::RGBA32_Float))
            {
                grad_derivative_barycentric = GpuTexture2D(
                    gpu_system_, width, height, 1, GpuFormat::RGBA32_Float, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
            }
            grad_barycentric.Name("GpuDiffRender.InterpolateBwd.grad_derivative_barycentric");
        }
        else
        {
            grad_derivative_barycentric = GpuTexture2D();
        }

        constexpr uint32_t BlockDim = 16;

        GpuConstantBufferOfType<InterpolateBwdConstantBuffer> interpolate_bwd_cb(gpu_system_, "interpolate_bwd_cb");
        interpolate_bwd_cb->gbuffer_size = glm::uvec2(width, height);
        interpolate_bwd_cb->num_attribs = num_attribs_per_vtx;
        interpolate_bwd_cb.UploadStaging();

        const GpuConstantBufferView interpolate_bwd_cbv(gpu_system_, interpolate_bwd_cb);

        const GpuShaderResourceView barycentric_srv(gpu_system_, barycentric, GpuFormat::RG32_Float);
        const GpuShaderResourceView prim_id_srv(gpu_system_, prim_id, GpuFormat::R32_Uint);
        const GpuShaderResourceView vtx_attribs_srv(gpu_system_, vtx_attribs, GpuFormat::R32_Float);
        const GpuShaderResourceView indices_srv(gpu_system_, indices, GpuFormat::R32_Uint);
        const GpuShaderResourceView grad_shading_srv(gpu_system_, grad_shading, GpuFormat::R32_Float);
        GpuShaderResourceView derivative_barycentric_srv;
        GpuShaderResourceView grad_derivative_shading_srv;
        if (needs_dbc)
        {
            derivative_barycentric_srv = GpuShaderResourceView(gpu_system_, derivative_barycentric, GpuFormat::RGBA32_Float);
            grad_derivative_shading_srv = GpuShaderResourceView(gpu_system_, grad_derivative_shading, GpuFormat::RG32_Float);
        }

        GpuUnorderedAccessView grad_vtx_attribs_uav(gpu_system_, grad_vtx_attribs, GpuFormat::R32_Uint);
        GpuUnorderedAccessView grad_barycentric_uav(gpu_system_, grad_barycentric, GpuFormat::RG32_Float);
        GpuUnorderedAccessView grad_derivative_barycentric_uav;
        if (needs_dbc)
        {
            grad_derivative_barycentric_uav = GpuUnorderedAccessView(gpu_system_, grad_derivative_barycentric, GpuFormat::RGBA32_Float);
        }

        {
            const uint32_t clear_clr[] = {0, 0, 0, 0};
            cmd_list.Clear(grad_vtx_attribs_uav, clear_clr);
        }
        {
            const float clear_clr[] = {0, 0, 0, 0};
            cmd_list.Clear(grad_barycentric_uav, clear_clr);
            if (needs_dbc)
            {
                cmd_list.Clear(grad_derivative_barycentric_uav, clear_clr);
            }
        }

        std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
            {"param_cb", &interpolate_bwd_cbv},
        };
        std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
            {"barycentric_tex", &barycentric_srv},
            {"prim_id_tex", &prim_id_srv},
            {"vtx_attribs_buff", &vtx_attribs_srv},
            {"indices_buff", &indices_srv},
            {"grad_shading_buff", &grad_shading_srv},
            {"derivative_barycentric_tex", &derivative_barycentric_srv},
            {"grad_derivative_shading_buff", &grad_derivative_shading_srv},
        };
        std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
            {"grad_vtx_attribs", &grad_vtx_attribs_uav},
            {"grad_barycentric", &grad_barycentric_uav},
            {"grad_derivative_barycentric", &grad_derivative_barycentric_uav},
        };
        const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
        cmd_list.Compute(needs_dbc ? interpolate_bwd_derivative_attribs_pipeline_ : interpolate_bwd_pipeline_, DivUp(width, BlockDim),
            DivUp(height, BlockDim), 1, shader_binding);
    }

    void GpuDiffRender::GenerateMipmaps(GpuCommandList& cmd_list, GpuTexture2D& texture, uint32_t mip_levels)
    {
        const uint32_t width = texture.Width(0);
        const uint32_t height = texture.Height(0);

        if (mip_levels == 0)
        {
            mip_levels = LogNextPowerOf2(std::max(texture.Width(0), texture.Height(0)));
        }

        GpuTexture2D* texture_mip;
        GpuTexture2D texture_temp;
        if ((texture.Format() != GpuFormat::RGBA32_Float) || (texture.MipLevels() != mip_levels))
        {
            texture_temp = GpuTexture2D(gpu_system_, width, height, mip_levels, GpuFormat::RGBA32_Float,
                GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, "GpuDiffRender.GenerateMipmap.texture_mip");
            texture_mip = &texture_temp;

            if (texture.Format() != GpuFormat::RGBA32_Float)
            {
                constexpr uint32_t BlockDim = 16;

                GpuConstantBufferOfType<TextureCopyConstantBuffer> texture_copy_cb(gpu_system_, "texture_copy_cb");
                texture_copy_cb->tex_size = glm::uvec2(width, height);
                texture_copy_cb.UploadStaging();

                const GpuConstantBufferView texture_copy_cbv(gpu_system_, texture_copy_cb);

                const GpuShaderResourceView texture_srv(gpu_system_, texture, 0);

                GpuUnorderedAccessView texture_mip_uav(gpu_system_, *texture_mip, 0);

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &texture_copy_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"texture", &texture_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"texture_f32", &texture_mip_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(texture_copy_pipeline_, DivUp(width, BlockDim), DivUp(height, BlockDim), 1, shader_binding);
            }
            else
            {
                cmd_list.Copy(texture_temp, 0, 0, 0, 0, texture, 0, GpuBox{0, 0, 0, width, height, 1});
            }
        }
        else
        {
            texture_mip = &texture;
        }

        cmd_list.GenerateMipmaps(*texture_mip, GpuSampler::Filter::Linear);

        if (&texture != texture_mip)
        {
            texture = std::move(*texture_mip);
        }
    }

    void GpuDiffRender::TextureFwd(GpuCommandList& cmd_list, const GpuTexture2D& texture, const GpuTexture2D& prim_id, const GpuBuffer& uv,
        const GpuBuffer& derivative_uv, const GpuDynamicSampler& sampler, GpuTexture2D& image)
    {
        const uint32_t gbuffer_width = prim_id.Width(0);
        const uint32_t gbuffer_height = prim_id.Height(0);
        const uint32_t mip_levels = texture.MipLevels();
        const bool mip_mode = static_cast<bool>(derivative_uv) && (mip_levels > 1);

        if ((image.Width(0) != gbuffer_width) || (image.Height(0) != gbuffer_height) || (image.Format() != texture.Format()))
        {
            image = GpuTexture2D(gpu_system_, gbuffer_width, gbuffer_height, 1, texture.Format(),
                GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        image.Name("GpuDiffRender.TextureFwd.image");

        constexpr uint32_t BlockDim = 16;

        GpuConstantBufferOfType<TextureFwdConstantBuffer> texture_fwd_cb(gpu_system_, "texture_fwd_cb");
        texture_fwd_cb->gbuffer_size = glm::uvec2(gbuffer_width, gbuffer_height);
        texture_fwd_cb->tex_size = glm::uvec2(texture.Width(0), texture.Height(0));
        texture_fwd_cb->mip_levels = mip_levels;
        texture_fwd_cb.UploadStaging();

        const GpuConstantBufferView texture_fwd_cbv(gpu_system_, texture_fwd_cb);

        const GpuShaderResourceView prim_id_srv(gpu_system_, prim_id, GpuFormat::R32_Uint);
        const GpuShaderResourceView texture_srv(gpu_system_, texture);
        const GpuShaderResourceView uv_srv(gpu_system_, uv, GpuFormat::RG32_Float);
        GpuShaderResourceView derivative_uv_srv;
        if (mip_mode)
        {
            derivative_uv_srv = GpuShaderResourceView(gpu_system_, derivative_uv, GpuFormat::RGBA32_Float);
        }

        GpuUnorderedAccessView image_uav(gpu_system_, image);

        const float clear_clr[] = {0, 0, 0, 0};
        cmd_list.Clear(image_uav, clear_clr);

        std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
            {"param_cb", &texture_fwd_cbv},
        };
        std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
            {"prim_id_tex", &prim_id_srv},
            {"texture", &texture_srv},
            {"uv_buff", &uv_srv},
            {"derivative_uv_buff", &derivative_uv_srv},
        };
        std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
            {"image", &image_uav},
        };
        std::tuple<std::string_view, const GpuDynamicSampler*> samplers[] = {
            {"tex_sampler", &sampler},
        };
        const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs, samplers};
        cmd_list.Compute(mip_mode ? texture_fwd_mip_pipeline_ : texture_fwd_pipeline_, DivUp(gbuffer_width, BlockDim),
            DivUp(gbuffer_height, BlockDim), 1, shader_binding);
    }

    void GpuDiffRender::TextureBwd(GpuCommandList& cmd_list, const GpuTexture2D& texture, const GpuTexture2D& prim_id, const GpuBuffer& uv,
        const GpuBuffer& derivative_uv, const GpuTexture2D& grad_image, const GpuDynamicSampler& sampler, GpuBuffer& grad_texture,
        GpuBuffer& grad_uv, GpuBuffer& grad_derivative_uv)
    {
        const uint32_t gbuffer_width = prim_id.Width(0);
        const uint32_t gbuffer_height = prim_id.Height(0);
        const uint32_t tex_width = texture.Width(0);
        const uint32_t tex_height = texture.Height(0);
        const uint32_t mip_levels = texture.MipLevels();
        const uint32_t num_channels = FormatChannels(texture.Format());
        const bool mip_mode = static_cast<bool>(derivative_uv) && (mip_levels > 1);

        auto mip_level_offsets = std::make_unique<uint32_t[]>(mip_levels + 1);
        mip_level_offsets[0] = 0;
        for (uint32_t i = 1; i <= mip_levels; ++i)
        {
            const uint32_t last_level_size = texture.Width(i - 1) * texture.Height(i - 1) * num_channels;
            mip_level_offsets[i] = mip_level_offsets[i - 1] + last_level_size;
        }

        GpuBuffer grad_texture_mips;
        if (mip_mode)
        {
            grad_texture_mips =
                GpuBuffer(gpu_system_, mip_level_offsets[mip_levels] * sizeof(float), GpuHeap::Default, GpuResourceFlag::UnorderedAccess);
        }

        const uint32_t grad_texture_size = mip_level_offsets[1] * sizeof(float);
        if (grad_texture.Size() != grad_texture_size)
        {
            grad_texture =
                GpuBuffer(gpu_system_, grad_texture_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_texture.Name("GpuDiffRender.TextureBwd.grad_texture");

        const uint32_t grad_uv_size = gbuffer_width * gbuffer_height * sizeof(glm::vec2);
        if (grad_uv.Size() != grad_uv_size)
        {
            grad_uv = GpuBuffer(gpu_system_, grad_uv_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_uv.Name("GpuDiffRender.TextureBwd.grad_uv");

        if (mip_mode)
        {
            const uint32_t grad_derivative_uv_size = tex_width * tex_height * sizeof(glm::vec4);
            if (grad_derivative_uv.Size() != grad_derivative_uv_size)
            {
                grad_derivative_uv = GpuBuffer(
                    gpu_system_, grad_derivative_uv_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
            }
            grad_derivative_uv.Name("GpuDiffRender.TextureBwd.grad_derivative_uv");
        }
        else
        {
            grad_derivative_uv = GpuBuffer();
        }

        {
            constexpr uint32_t BlockDim = 16;

            GpuConstantBufferOfType<TextureBwdConstantBuffer> texture_bwd_cb(gpu_system_, "texture_bwd_cb");
            texture_bwd_cb->gbuffer_size = glm::uvec2(gbuffer_width, gbuffer_height);
            texture_bwd_cb->tex_size = glm::uvec2(tex_width, tex_height);
            texture_bwd_cb->num_channels = num_channels;
            texture_bwd_cb->min_mag_filter_linear = sampler.SamplerFilters().min == GpuSampler::Filter::Linear;
            texture_bwd_cb->mip_filter_linear = sampler.SamplerFilters().mip == GpuSampler::Filter::Linear;
            texture_bwd_cb->mip_levels = mip_levels;
            texture_bwd_cb->address_mode = static_cast<uint32_t>(sampler.SamplerAddressModes().u);
            std::memcpy(texture_bwd_cb->mip_level_offsets, mip_level_offsets.get(),
                std::min<uint32_t>(mip_levels * sizeof(uint32_t), sizeof(texture_bwd_cb->mip_level_offsets)));
            texture_bwd_cb.UploadStaging();

            const GpuConstantBufferView texture_bwd_cbv(gpu_system_, texture_bwd_cb);

            const GpuShaderResourceView texture_srv(gpu_system_, texture);
            const GpuShaderResourceView uv_srv(gpu_system_, uv, GpuFormat::RG32_Float);
            const GpuShaderResourceView grad_image_srv(gpu_system_, grad_image);
            GpuShaderResourceView derivative_uv_srv;
            if (mip_mode)
            {
                derivative_uv_srv = GpuShaderResourceView(gpu_system_, derivative_uv, GpuFormat::RGBA32_Float);
            }

            GpuUnorderedAccessView grad_texture_mips_uav(gpu_system_, mip_mode ? grad_texture_mips : grad_texture,
                GpuFormat::R32_Uint); // Float as uint due to atomic operations
            GpuUnorderedAccessView grad_uv_uav(gpu_system_, grad_uv, GpuFormat::RG32_Float);
            GpuUnorderedAccessView grad_derivative_uv_uav;
            if (mip_mode)
            {
                grad_derivative_uv_uav = GpuUnorderedAccessView(gpu_system_, grad_derivative_uv, GpuFormat::RGBA32_Float);
            }

            {
                const uint32_t clear_clr[] = {0, 0, 0, 0};
                cmd_list.Clear(grad_texture_mips_uav, clear_clr);
            }
            {
                const float clear_clr[] = {0, 0, 0, 0};
                cmd_list.Clear(grad_uv_uav, clear_clr);
                if (mip_mode)
                {
                    cmd_list.Clear(grad_derivative_uv_uav, clear_clr);
                }
            }

            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                {"param_cb", &texture_bwd_cbv},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"texture", &texture_srv},
                {"uv_buff", &uv_srv},
                {"grad_image", &grad_image_srv},
                {"derivative_uv_buff", &derivative_uv_srv},
            };
            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                {"grad_texture", &grad_texture_mips_uav},
                {"grad_uv", &grad_uv_uav},
                {"grad_derivative_uv", &grad_derivative_uv_uav},
            };
            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
            cmd_list.Compute(mip_mode ? texture_bwd_mip_pipeline_ : texture_bwd_pipeline_, DivUp(gbuffer_width, BlockDim),
                DivUp(gbuffer_height, BlockDim), 1, shader_binding);
        }

        if (mip_mode)
        {
            constexpr uint32_t BlockDim = 16;

            GpuConstantBufferOfType<AccumGradMipsConstantBuffer> accum_grad_mips_cb(gpu_system_, "accum_grad_mips_cb");
            accum_grad_mips_cb->tex_size = glm::uvec2(tex_width, tex_height);
            accum_grad_mips_cb->num_channels = num_channels;
            accum_grad_mips_cb->mip_levels = mip_levels;
            std::memcpy(accum_grad_mips_cb->mip_level_offsets, mip_level_offsets.get(),
                std::min<uint32_t>(mip_levels * sizeof(uint32_t), sizeof(accum_grad_mips_cb->mip_level_offsets)));
            accum_grad_mips_cb.UploadStaging();

            const GpuConstantBufferView accum_grad_mips_cbv(gpu_system_, accum_grad_mips_cb);

            const GpuShaderResourceView grad_texture_mips_srv(gpu_system_, grad_texture_mips, GpuFormat::R32_Float);

            GpuUnorderedAccessView grad_texture_uav(gpu_system_, grad_texture, GpuFormat::R32_Float);

            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                {"param_cb", &accum_grad_mips_cbv},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"grad_texture_mips", &grad_texture_mips_srv},
            };
            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                {"grad_texture", &grad_texture_uav},
            };
            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
            cmd_list.Compute(accum_grad_mips_pipeline_, DivUp(tex_width, BlockDim), DivUp(tex_height, BlockDim), 1, shader_binding);
        }
    }

    void GpuDiffRender::AntiAliasConstructOppositeVertices(GpuCommandList& cmd_list, const GpuBuffer& indices, GpuBuffer& opposite_vertices)
    {
        const uint32_t num_indices = indices.Size() / sizeof(uint32_t);

        const GpuShaderResourceView indices_srv(gpu_system_, indices, GpuFormat::R32_Uint);

        const uint32_t expected_hash_buff_size = num_indices * 2 * sizeof(glm::uvec3);
        if (opposite_vertices_hash_.Size() != expected_hash_buff_size)
        {
            opposite_vertices_hash_ = GpuBuffer(gpu_system_, expected_hash_buff_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess);
            opposite_vertices_hash_srv_ = GpuShaderResourceView(gpu_system_, opposite_vertices_hash_, GpuFormat::R32_Uint);
            opposite_vertices_hash_uav_ = GpuUnorderedAccessView(gpu_system_, opposite_vertices_hash_, GpuFormat::R32_Uint);
        }
        opposite_vertices_hash_.Name("GpuDiffRender.AntiAliasConstructOppositeVertices.opposite_vertices_hash");

        {
            constexpr uint32_t BlockDim = 256;

            GpuConstantBufferOfType<AntiAliasConstructOppositeVerticesHashConstantBuffer> anti_alias_construct_oppo_vert_hash_cb(
                gpu_system_, "anti_alias_construct_oppo_vert_hash_cb");
            anti_alias_construct_oppo_vert_hash_cb->num_indices = num_indices;
            anti_alias_construct_oppo_vert_hash_cb->hash_size = opposite_vertices_hash_.Size() / sizeof(glm::uvec3);
            anti_alias_construct_oppo_vert_hash_cb.UploadStaging();

            const GpuConstantBufferView anti_alias_construct_oppo_vert_hash_cbv(gpu_system_, anti_alias_construct_oppo_vert_hash_cb);

            {
                const uint32_t clear_clr[] = {~0U, ~0U, ~0U, ~0U};
                cmd_list.Clear(opposite_vertices_hash_uav_, clear_clr);
            }

            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                {"param_cb", &anti_alias_construct_oppo_vert_hash_cbv},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"indices_buff", &indices_srv},
            };
            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                {"hash_table", &opposite_vertices_hash_uav_},
            };
            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
            cmd_list.Compute(anti_alias_construct_oppo_vert_hash_pipeline_, DivUp(num_indices, BlockDim), 1, 1, shader_binding);
        }

        if (opposite_vertices.Size() != indices.Size())
        {
            opposite_vertices = GpuBuffer(gpu_system_, indices.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess);
        }
        opposite_vertices.Name("GpuDiffRender.AntiAliasConstructOppositeVertices.opposite_vertices");

        {
            constexpr uint32_t BlockDim = 256;

            GpuConstantBufferOfType<AntiAliasConstructOppositeVerticesConstantBuffer> anti_alias_construct_oppo_vert_cb(
                gpu_system_, "anti_alias_construct_oppo_vert_cb");
            anti_alias_construct_oppo_vert_cb->num_indices = num_indices;
            anti_alias_construct_oppo_vert_cb->hash_size = opposite_vertices_hash_.Size() / sizeof(glm::uvec3);
            anti_alias_construct_oppo_vert_cb.UploadStaging();

            const GpuConstantBufferView anti_alias_construct_oppo_vert_cbv(gpu_system_, anti_alias_construct_oppo_vert_cb);

            GpuUnorderedAccessView oppo_vert_uav(gpu_system_, opposite_vertices, GpuFormat::R32_Uint);

            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                {"param_cb", &anti_alias_construct_oppo_vert_cbv},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"indices_buff", &indices_srv},
                {"hash_table", &opposite_vertices_hash_srv_},
            };
            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                {"oppo_vert", &oppo_vert_uav},
            };
            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
            cmd_list.Compute(anti_alias_construct_oppo_vert_pipeline_, DivUp(num_indices, BlockDim), 1, 1, shader_binding);
        }
    }

    void GpuDiffRender::AntiAliasFwd(GpuCommandList& cmd_list, const GpuBuffer& shading, const GpuTexture2D& prim_id,
        const GpuBuffer& positions, const GpuBuffer& indices, const GpuViewport& viewport, const GpuBuffer& opposite_vertices,
        GpuBuffer& anti_aliased)
    {
        const uint32_t width = prim_id.Width(0);
        const uint32_t height = prim_id.Height(0);
        const uint32_t num_attribs = shading.Size() / (width * height * sizeof(float));

        if (anti_aliased.Size() != shading.Size())
        {
            anti_aliased =
                GpuBuffer(gpu_system_, shading.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        anti_aliased.Name("GpuDiffRender.AntiAliasFwd.anti_aliased");

        const uint32_t silhouette_info_size = width * height * 4 * sizeof(uint32_t);
        if (silhouette_info_.Size() != silhouette_info_size)
        {
            silhouette_info_ = GpuBuffer(gpu_system_, silhouette_info_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess,
                "GpuDiffRender.AntiAliasFwd.silhouette_info");
            silhouette_info_srv_ = GpuShaderResourceView(gpu_system_, silhouette_info_, GpuFormat::R32_Uint);
            silhouette_info_uav_ = GpuUnorderedAccessView(gpu_system_, silhouette_info_, GpuFormat::R32_Uint);
        }

        cmd_list.Copy(anti_aliased, shading);

        {
            constexpr uint32_t BlockDim = 16;

            GpuConstantBufferOfType<AntiAliasFwdConstantBuffer> anti_alias_fwd_cb(gpu_system_, "anti_alias_fwd_cb");
            anti_alias_fwd_cb->viewport = glm::vec4(viewport.left, viewport.top, viewport.width, viewport.height);
            anti_alias_fwd_cb->gbuffer_size = glm::uvec2(width, height);
            anti_alias_fwd_cb->num_attribs = num_attribs;
            anti_alias_fwd_cb.UploadStaging();

            const GpuConstantBufferView anti_alias_fwd_cbv(gpu_system_, anti_alias_fwd_cb);

            const GpuShaderResourceView shading_srv(gpu_system_, shading, GpuFormat::R32_Float);
            const GpuShaderResourceView prim_id_srv(gpu_system_, prim_id, GpuFormat::R32_Uint);
            const GpuShaderResourceView positions_srv(gpu_system_, positions, GpuFormat::RGBA32_Float);
            const GpuShaderResourceView indices_srv(gpu_system_, indices, GpuFormat::R32_Uint);
            const GpuShaderResourceView opposite_vertices_srv(gpu_system_, opposite_vertices, GpuFormat::R32_Uint);

            GpuUnorderedAccessView anti_aliased_uav(gpu_system_, anti_aliased, GpuFormat::R32_Uint);

            {
                const uint32_t clear_clr[] = {0, 0, 0, 0};
                cmd_list.Clear(silhouette_counter_uav_, clear_clr);
            }

            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                {"param_cb", &anti_alias_fwd_cbv},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"shading_buff", &shading_srv},
                {"prim_id_tex", &prim_id_srv},
                {"depth_tex", &depth_srv_},
                {"positions_buff", &positions_srv},
                {"indices_buff", &indices_srv},
                {"opposite_vertices_buff", &opposite_vertices_srv},
            };
            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                {"anti_aliased", &anti_aliased_uav},
                {"silhouette_counter", &silhouette_counter_uav_},
                {"silhouette_info", &silhouette_info_uav_},
            };
            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
            cmd_list.Compute(anti_alias_fwd_pipeline_, DivUp(width, BlockDim), DivUp(height, BlockDim), 1, shader_binding);
        }
    }

    void GpuDiffRender::AntiAliasBwd(GpuCommandList& cmd_list, const GpuBuffer& shading, const GpuTexture2D& prim_id,
        const GpuBuffer& positions, const GpuBuffer& indices, const GpuViewport& viewport, const GpuBuffer& grad_anti_aliased,
        GpuBuffer& grad_shading, GpuBuffer& grad_positions)
    {
        const uint32_t width = prim_id.Width(0);
        const uint32_t height = prim_id.Height(0);
        const uint32_t num_attribs = shading.Size() / (width * height * sizeof(float));

        {
            // constexpr uint32_t BlockDim = 32;

            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                {"param_cb", &anti_alias_indirect_args_cbv_},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"silhouette_counter", &silhouette_counter_srv_},
            };
            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                {"indirect_args", &indirect_args_uav_},
            };
            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
            cmd_list.Compute(anti_alias_indirect_pipeline_, 1, 1, 1, shader_binding);
        }

        if (grad_shading.Size() != shading.Size())
        {
            grad_shading =
                GpuBuffer(gpu_system_, shading.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_shading.Name("GpuDiffRender.AntiAliasBwd.grad_shading");

        if (grad_positions.Size() != positions.Size())
        {
            grad_positions =
                GpuBuffer(gpu_system_, positions.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_positions.Name("GpuDiffRender.AntiAliasBwd.grad_positions");

        cmd_list.Copy(grad_shading, grad_anti_aliased);

        {
            // constexpr uint32_t BlockDim = 256;

            GpuConstantBufferOfType<AntiAliasBwdConstantBuffer> anti_alias_bwd_cb(gpu_system_, "anti_alias_bwd_cb");
            anti_alias_bwd_cb->viewport = glm::vec4(viewport.left, viewport.top, viewport.width, viewport.height);
            anti_alias_bwd_cb->gbuffer_size = glm::uvec2(width, height);
            anti_alias_bwd_cb->num_attribs = num_attribs;
            anti_alias_bwd_cb.UploadStaging();

            const GpuConstantBufferView anti_alias_bwd_cbv(gpu_system_, anti_alias_bwd_cb);

            const GpuShaderResourceView shading_srv(gpu_system_, shading, GpuFormat::R32_Float);
            const GpuShaderResourceView prim_id_srv(gpu_system_, prim_id, GpuFormat::R32_Uint);
            const GpuShaderResourceView positions_srv(gpu_system_, positions, GpuFormat::RGBA32_Float);
            const GpuShaderResourceView indices_srv(gpu_system_, indices, GpuFormat::R32_Uint);
            const GpuShaderResourceView grad_anti_aliased_srv(gpu_system_, grad_anti_aliased, GpuFormat::R32_Float);

            GpuUnorderedAccessView grad_shading_uav(gpu_system_, grad_shading, GpuFormat::R32_Uint);
            GpuUnorderedAccessView grad_positions_uav(gpu_system_, grad_positions, GpuFormat::R32_Uint);

            {
                const uint32_t clear_clr[] = {0, 0, 0, 0};
                cmd_list.Clear(grad_positions_uav, clear_clr);
            }

            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                {"param_cb", &anti_alias_bwd_cbv},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"shading_buff", &shading_srv},
                {"prim_id_tex", &prim_id_srv},
                {"positions_buff", &positions_srv},
                {"indices_buff", &indices_srv},
                {"silhouette_counter", &silhouette_counter_srv_},
                {"silhouette_info", &silhouette_info_srv_},
                {"grad_anti_aliased", &grad_anti_aliased_srv},
            };
            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                {"grad_shading", &grad_shading_uav},
                {"grad_positions", &grad_positions_uav},
            };
            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
            cmd_list.ComputeIndirect(anti_alias_bwd_pipeline_, indirect_args_, shader_binding);
        }
    }
} // namespace AIHoloImager
