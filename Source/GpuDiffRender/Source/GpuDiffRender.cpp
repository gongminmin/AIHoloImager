// Copyright (c) 2025 Minmin Gong
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

#include "CompiledShader/AntiAliasConstructOppoVertCs.h"
#include "CompiledShader/AntiAliasConstructOppoVertHashCs.h"
#include "CompiledShader/AntialiasBwdCs.h"
#include "CompiledShader/AntialiasFwdCs.h"
#include "CompiledShader/AntialiasIndirectCs.h"
#include "CompiledShader/InterpolateBwdCs.h"
#include "CompiledShader/InterpolateFwdCs.h"
#include "CompiledShader/RasterizeBwdCs.h"
#include "CompiledShader/RasterizeFwdGs.h"
#include "CompiledShader/RasterizeFwdPs.h"
#include "CompiledShader/RasterizeFwdVs.h"
#include "CompiledShader/TextureBwdCs.h"
#include "CompiledShader/TextureCopyCs.h"
#include "CompiledShader/TextureFwdCs.h"

namespace AIHoloImager
{
    GpuDiffRender::GpuDiffRender(GpuSystem& gpu_system) : gpu_system_(gpu_system)
    {
        {
            const ShaderInfo shaders[] = {
                {RasterizeFwdVs_shader, 0, 0, 0},
                {RasterizeFwdPs_shader, 0, 0, 0},
                {RasterizeFwdGs_shader, 0, 0, 0},
            };

            const GpuFormat rtv_formats[] = {GpuFormat::RG32_Float, GpuFormat::R32_Uint};

            GpuRenderPipeline::States states;
            states.cull_mode = GpuRenderPipeline::CullMode::ClockWise;
            states.rtv_formats = rtv_formats;
            states.dsv_format = GpuFormat::D32_Float;
            states.depth_enable = true;

            const GpuVertexAttribs vertex_attribs(std::span<const GpuVertexAttrib>({
                {"POSITION", 0, GpuFormat::RGBA32_Float},
            }));

            rasterize_fwd_pipeline_ =
                GpuRenderPipeline(gpu_system_, GpuRenderPipeline::PrimitiveTopology::TriangleList, shaders, vertex_attribs, {}, states);
        }
        {
            const ShaderInfo shader = {RasterizeBwdCs_shader, 1, 5, 1};
            rasterize_bwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }

        {
            const ShaderInfo shader = {InterpolateFwdCs_shader, 1, 4, 1};
            interpolate_fwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {InterpolateBwdCs_shader, 1, 5, 2};
            interpolate_bwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }

        {
            const ShaderInfo shader = {AntiAliasConstructOppoVertHashCs_shader, 1, 1, 1};
            anti_alias_construct_oppo_vert_hash_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {AntiAliasConstructOppoVertCs_shader, 1, 2, 1};
            anti_alias_construct_oppo_vert_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            anti_alias_indirect_args_cb_ =
                GpuConstantBufferOfType<AntiAliasIndirectArgsConstantBuffer>(gpu_system_, L"anti_alias_indirect_args_cb_");
            anti_alias_indirect_args_cb_->bwd_block_dim = 256;
            anti_alias_indirect_args_cb_.UploadStaging();

            const ShaderInfo shader = {AntiAliasIndirectCs_shader, 1, 1, 1};
            anti_alias_indirect_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {AntiAliasFwdCs_shader, 1, 6, 3};
            anti_alias_fwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {AntiAliasBwdCs_shader, 1, 7, 2};
            anti_alias_bwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }

        silhouette_counter_ = GpuBuffer(gpu_system_, sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::UnorderedAccess,
            L"GpuDiffRender.AntiAliasFwd.silhouette_counter");
        silhouette_counter_srv_ = GpuShaderResourceView(gpu_system_, silhouette_counter_, GpuFormat::R32_Uint);
        silhouette_counter_uav_ = GpuUnorderedAccessView(gpu_system_, silhouette_counter_, GpuFormat::R32_Uint);

        indirect_args_ = GpuBuffer(gpu_system_, 3 * sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::UnorderedAccess,
            L"GpuDiffRender.AntiAliasBwd.indirect_args");
        indirect_args_uav_ = GpuUnorderedAccessView(gpu_system_, indirect_args_, GpuFormat::R32_Uint);

        {
            const ShaderInfo shader = {TextureCopyCs_shader, 1, 1, 1};
            texture_copy_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {TextureFwdCs_shader, 1, 3, 1, 1};
            texture_fwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
        {
            const ShaderInfo shader = {TextureBwdCs_shader, 1, 3, 2};
            texture_bwd_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
        }
    }

    GpuDiffRender::~GpuDiffRender() = default;

    void GpuDiffRender::RasterizeFwd(GpuCommandList& cmd_list, const GpuBuffer& positions, const GpuBuffer& indices, uint32_t width,
        uint32_t height, const GpuViewport& viewport, GpuTexture2D& barycentric, GpuTexture2D& prim_id)
    {
        if ((barycentric.Width(0) != width) || (barycentric.Height(0) != height) || (barycentric.Format() != GpuFormat::RG32_Float))
        {
            barycentric = GpuTexture2D(
                gpu_system_, width, height, 1, GpuFormat::RG32_Float, GpuResourceFlag::RenderTarget | GpuResourceFlag::Shareable);
        }
        barycentric.Name(L"GpuDiffRender.RasterizeFwd.barycentric");

        if ((prim_id.Width(0) != width) || (prim_id.Height(0) != height) || (prim_id.Format() != GpuFormat::R32_Uint))
        {
            prim_id = GpuTexture2D(
                gpu_system_, width, height, 1, GpuFormat::R32_Uint, GpuResourceFlag::RenderTarget | GpuResourceFlag::Shareable);
        }
        prim_id.Name(L"GpuDiffRender.RasterizeFwd.prim_id");

        GpuRenderTargetView barycentric_rtv(gpu_system_, barycentric);
        GpuRenderTargetView prim_id_rtv(gpu_system_, prim_id);

        if ((depth_tex_.Width(0) != width) || (depth_tex_.Height(0) != height))
        {
            depth_tex_ = GpuTexture2D(gpu_system_, width, height, 1, GpuFormat::D32_Float, GpuResourceFlag::DepthStencil,
                L"GpuDiffRender.RasterizeFwd.depth_tex");
            depth_srv_ = GpuShaderResourceView(gpu_system_, depth_tex_, GpuFormat::R32_Float);
            depth_dsv_ = GpuDepthStencilView(gpu_system_, depth_tex_);
        }

        const float clear_clr[] = {0, 0, 0, 0};
        cmd_list.Clear(prim_id_rtv, clear_clr);
        cmd_list.ClearDepth(depth_dsv_, 1.0f);

        const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&positions, 0, sizeof(glm::vec4)}};
        const GpuCommandList::IndexBufferBinding ib_binding = {&indices, 0, GpuFormat::R32_Uint};

        const GpuCommandList::ShaderBinding shader_bindings[] = {
            {{}, {}, {}},
            {{}, {}, {}},
            {{}, {}, {}},
        };

        const GpuRenderTargetView* rtvs[] = {&barycentric_rtv, &prim_id_rtv};

        cmd_list.Render(rasterize_fwd_pipeline_, vb_bindings, &ib_binding, indices.Size() / sizeof(uint32_t), shader_bindings, rtvs,
            &depth_dsv_, std::span(&viewport, 1), {});
    }

    void GpuDiffRender::RasterizeBwd(GpuCommandList& cmd_list, const GpuBuffer& positions, const GpuBuffer& indices,
        const GpuViewport& viewport, const GpuTexture2D& barycentric, const GpuTexture2D& prim_id, const GpuTexture2D& grad_barycentric,
        GpuBuffer& grad_positions)
    {
        const uint32_t width = barycentric.Width(0);
        const uint32_t height = barycentric.Height(0);

        if (grad_positions.Size() != positions.Size())
        {
            grad_positions =
                GpuBuffer(gpu_system_, positions.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_positions.Name(L"GpuDiffRender.RasterizeBwd.grad_positions");

        constexpr uint32_t BlockDim = 16;

        GpuConstantBufferOfType<RasterizeBwdConstantBuffer> rasterize_bwd_cb(gpu_system_, L"rasterize_bwd_cb");
        rasterize_bwd_cb->viewport = glm::vec4(viewport.left, viewport.top, viewport.width, viewport.height);
        rasterize_bwd_cb->gbuffer_size = glm::uvec2(width, height);
        rasterize_bwd_cb.UploadStaging();

        const GpuShaderResourceView barycentric_srv(gpu_system_, barycentric, GpuFormat::RG32_Float);
        const GpuShaderResourceView prim_id_srv(gpu_system_, prim_id, GpuFormat::R32_Uint);
        const GpuShaderResourceView grad_barycentric_srv(gpu_system_, grad_barycentric, GpuFormat::RG32_Float);
        const GpuShaderResourceView positions_srv(gpu_system_, positions, GpuFormat::RGBA32_Float);
        const GpuShaderResourceView indices_srv(gpu_system_, indices, GpuFormat::R32_Uint);

        GpuUnorderedAccessView grad_positions_uav(
            gpu_system_, grad_positions, GpuFormat::R32_Uint); // Float as uint due to atomic operations

        const uint32_t clear_clr[] = {0, 0, 0, 0};
        cmd_list.Clear(grad_positions_uav, clear_clr);

        const GpuConstantBuffer* cbs[] = {&rasterize_bwd_cb};
        const GpuShaderResourceView* srvs[] = {&barycentric_srv, &prim_id_srv, &grad_barycentric_srv, &positions_srv, &indices_srv};
        GpuUnorderedAccessView* uavs[] = {&grad_positions_uav};
        const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
        cmd_list.Compute(rasterize_bwd_pipeline_, DivUp(width, BlockDim), DivUp(height, BlockDim), 1, shader_binding);
    }

    void GpuDiffRender::InterpolateFwd(GpuCommandList& cmd_list, const GpuBuffer& vtx_attribs, uint32_t num_attribs_per_vtx,
        const GpuTexture2D& barycentric, const GpuTexture2D& prim_id, const GpuBuffer& indices, GpuBuffer& shading)
    {
        const uint32_t width = barycentric.Width(0);
        const uint32_t height = barycentric.Height(0);

        const uint32_t shading_size = width * height * num_attribs_per_vtx * sizeof(float);
        if (shading.Size() != shading_size)
        {
            shading = GpuBuffer(gpu_system_, shading_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        shading.Name(L"GpuDiffRender.InterpolateFwd.shading");

        constexpr uint32_t BlockDim = 16;

        GpuConstantBufferOfType<InterpolateFwdConstantBuffer> interpolate_fwd_cb(gpu_system_, L"interpolate_fwd_cb");
        interpolate_fwd_cb->gbuffer_size = glm::uvec2(width, height);
        interpolate_fwd_cb->num_attribs = num_attribs_per_vtx;
        interpolate_fwd_cb.UploadStaging();

        const GpuShaderResourceView barycentric_srv(gpu_system_, barycentric, GpuFormat::RG32_Float);
        const GpuShaderResourceView prim_id_srv(gpu_system_, prim_id, GpuFormat::R32_Uint);
        const GpuShaderResourceView attrib_srv(gpu_system_, vtx_attribs, GpuFormat::R32_Float);
        const GpuShaderResourceView index_srv(gpu_system_, indices, GpuFormat::R32_Uint);

        GpuUnorderedAccessView shading_uav(gpu_system_, shading, GpuFormat::R32_Float);

        const float clear_clr[] = {0, 0, 0, 0};
        cmd_list.Clear(shading_uav, clear_clr);

        const GpuConstantBuffer* cbs[] = {&interpolate_fwd_cb};
        const GpuShaderResourceView* srvs[] = {&barycentric_srv, &prim_id_srv, &attrib_srv, &index_srv};
        GpuUnorderedAccessView* uavs[] = {&shading_uav};
        const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
        cmd_list.Compute(interpolate_fwd_pipeline_, DivUp(width, BlockDim), DivUp(height, BlockDim), 1, shader_binding);
    }

    void GpuDiffRender::InterpolateBwd(GpuCommandList& cmd_list, const GpuBuffer& vtx_attribs, uint32_t num_attribs_per_vtx,
        const GpuTexture2D& barycentric, const GpuTexture2D& prim_id, const GpuBuffer& indices, const GpuBuffer& grad_shading,
        GpuBuffer& grad_vtx_attribs, GpuTexture2D& grad_barycentric)
    {
        const uint32_t width = barycentric.Width(0);
        const uint32_t height = barycentric.Height(0);

        if (grad_vtx_attribs.Size() != vtx_attribs.Size())
        {
            grad_vtx_attribs =
                GpuBuffer(gpu_system_, vtx_attribs.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_vtx_attribs.Name(L"GpuDiffRender.InterpolateBwd.grad_vtx_attribs");

        if ((grad_barycentric.Width(0) != width) || (grad_barycentric.Height(0) != height) ||
            (grad_barycentric.Format() != GpuFormat::RG32_Float))
        {
            grad_barycentric = GpuTexture2D(
                gpu_system_, width, height, 1, GpuFormat::RG32_Float, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_barycentric.Name(L"GpuDiffRender.InterpolateBwd.grad_barycentric");

        constexpr uint32_t BlockDim = 16;

        GpuConstantBufferOfType<InterpolateBwdConstantBuffer> interpolate_bwd_cb(gpu_system_, L"interpolate_bwd_cb");
        interpolate_bwd_cb->gbuffer_size = glm::uvec2(width, height);
        interpolate_bwd_cb->num_attribs = num_attribs_per_vtx;
        interpolate_bwd_cb.UploadStaging();

        const GpuShaderResourceView barycentric_srv(gpu_system_, barycentric, GpuFormat::RG32_Float);
        const GpuShaderResourceView prim_id_srv(gpu_system_, prim_id, GpuFormat::R32_Uint);
        const GpuShaderResourceView vtx_attribs_srv(gpu_system_, vtx_attribs, GpuFormat::R32_Float);
        const GpuShaderResourceView indices_srv(gpu_system_, indices, GpuFormat::R32_Uint);
        const GpuShaderResourceView grad_shading_srv(gpu_system_, grad_shading, GpuFormat::R32_Float);

        GpuUnorderedAccessView grad_vtx_attribs_uav(gpu_system_, grad_vtx_attribs, GpuFormat::R32_Uint);
        GpuUnorderedAccessView grad_barycentric_uav(gpu_system_, grad_barycentric, GpuFormat::RG32_Float);

        {
            const uint32_t clear_clr[] = {0, 0, 0, 0};
            cmd_list.Clear(grad_vtx_attribs_uav, clear_clr);
        }
        {
            const float clear_clr[] = {0, 0, 0, 0};
            cmd_list.Clear(grad_barycentric_uav, clear_clr);
        }

        const GpuConstantBuffer* cbs[] = {&interpolate_bwd_cb};
        const GpuShaderResourceView* srvs[] = {&barycentric_srv, &prim_id_srv, &vtx_attribs_srv, &indices_srv, &grad_shading_srv};
        GpuUnorderedAccessView* uavs[] = {&grad_vtx_attribs_uav, &grad_barycentric_uav};
        const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
        cmd_list.Compute(interpolate_bwd_pipeline_, DivUp(width, BlockDim), DivUp(height, BlockDim), 1, shader_binding);
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
        opposite_vertices_hash_.Name(L"GpuDiffRender.AntiAliasConstructOppositeVertices.opposite_vertices_hash");

        {
            constexpr uint32_t BlockDim = 256;

            GpuConstantBufferOfType<AntiAliasConstructOppositeVerticesHashConstantBuffer> anti_alias_construct_oppo_vert_hash_cb(
                gpu_system_, L"anti_alias_construct_oppo_vert_hash_cb");
            anti_alias_construct_oppo_vert_hash_cb->num_indices = num_indices;
            anti_alias_construct_oppo_vert_hash_cb->hash_size = opposite_vertices_hash_.Size() / sizeof(glm::uvec3);
            anti_alias_construct_oppo_vert_hash_cb.UploadStaging();

            {
                const uint32_t clear_clr[] = {~0U, ~0U, ~0U, ~0U};
                cmd_list.Clear(opposite_vertices_hash_uav_, clear_clr);
            }

            const GpuConstantBuffer* cbs[] = {&anti_alias_construct_oppo_vert_hash_cb};
            const GpuShaderResourceView* srvs[] = {&indices_srv};
            GpuUnorderedAccessView* uavs[] = {&opposite_vertices_hash_uav_};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(anti_alias_construct_oppo_vert_hash_pipeline_, DivUp(num_indices, BlockDim), 1, 1, shader_binding);
        }

        if (opposite_vertices.Size() != indices.Size())
        {
            opposite_vertices = GpuBuffer(gpu_system_, indices.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess);
        }
        opposite_vertices.Name(L"GpuDiffRender.AntiAliasConstructOppositeVertices.opposite_vertices");

        {
            constexpr uint32_t BlockDim = 256;

            GpuConstantBufferOfType<AntiAliasConstructOppositeVerticesConstantBuffer> anti_alias_construct_oppo_vert_cb(
                gpu_system_, L"anti_alias_construct_oppo_vert_cb");
            anti_alias_construct_oppo_vert_cb->num_indices = num_indices;
            anti_alias_construct_oppo_vert_cb->hash_size = opposite_vertices_hash_.Size() / sizeof(glm::uvec3);
            anti_alias_construct_oppo_vert_cb.UploadStaging();

            GpuUnorderedAccessView oppo_vert_uav(gpu_system_, opposite_vertices, GpuFormat::R32_Uint);

            const GpuConstantBuffer* cbs[] = {&anti_alias_construct_oppo_vert_cb};
            const GpuShaderResourceView* srvs[] = {&indices_srv, &opposite_vertices_hash_srv_};
            GpuUnorderedAccessView* uavs[] = {&oppo_vert_uav};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
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
        anti_aliased.Name(L"GpuDiffRender.AntiAliasFwd.anti_aliased");

        const uint32_t silhouette_info_size = width * height * 4 * sizeof(uint32_t);
        if (silhouette_info_.Size() != silhouette_info_size)
        {
            silhouette_info_ = GpuBuffer(gpu_system_, silhouette_info_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess,
                L"GpuDiffRender.AntiAliasFwd.silhouette_info");
            silhouette_info_srv_ = GpuShaderResourceView(gpu_system_, silhouette_info_, GpuFormat::R32_Uint);
            silhouette_info_uav_ = GpuUnorderedAccessView(gpu_system_, silhouette_info_, GpuFormat::R32_Uint);
        }

        cmd_list.Copy(anti_aliased, shading);

        {
            constexpr uint32_t BlockDim = 16;

            GpuConstantBufferOfType<AntiAliasFwdConstantBuffer> anti_alias_fwd_cb(gpu_system_, L"anti_alias_fwd_cb");
            anti_alias_fwd_cb->viewport = glm::vec4(viewport.left, viewport.top, viewport.width, viewport.height);
            anti_alias_fwd_cb->gbuffer_size = glm::uvec2(width, height);
            anti_alias_fwd_cb->num_attribs = num_attribs;
            anti_alias_fwd_cb.UploadStaging();

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

            const GpuConstantBuffer* cbs[] = {&anti_alias_fwd_cb};
            const GpuShaderResourceView* srvs[] = {
                &shading_srv, &prim_id_srv, &depth_srv_, &positions_srv, &indices_srv, &opposite_vertices_srv};
            GpuUnorderedAccessView* uavs[] = {&anti_aliased_uav, &silhouette_counter_uav_, &silhouette_info_uav_};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
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

            const GpuConstantBuffer* cbs[] = {&anti_alias_indirect_args_cb_};
            const GpuShaderResourceView* srvs[] = {&silhouette_counter_srv_};
            GpuUnorderedAccessView* uavs[] = {&indirect_args_uav_};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(anti_alias_indirect_pipeline_, 1, 1, 1, shader_binding);
        }

        if (grad_shading.Size() != shading.Size())
        {
            grad_shading =
                GpuBuffer(gpu_system_, shading.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_shading.Name(L"GpuDiffRender.AntiAliasBwd.grad_shading");

        if (grad_positions.Size() != positions.Size())
        {
            grad_positions =
                GpuBuffer(gpu_system_, positions.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_positions.Name(L"GpuDiffRender.AntiAliasBwd.grad_positions");

        cmd_list.Copy(grad_shading, grad_anti_aliased);

        {
            // constexpr uint32_t BlockDim = 256;

            GpuConstantBufferOfType<AntiAliasBwdConstantBuffer> anti_alias_bwd_cb(gpu_system_, L"anti_alias_bwd_cb");
            anti_alias_bwd_cb->viewport = glm::vec4(viewport.left, viewport.top, viewport.width, viewport.height);
            anti_alias_bwd_cb->gbuffer_size = glm::uvec2(width, height);
            anti_alias_bwd_cb->num_attribs = num_attribs;
            anti_alias_bwd_cb.UploadStaging();

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

            const GpuConstantBuffer* cbs[] = {&anti_alias_bwd_cb};
            const GpuShaderResourceView* srvs[] = {&shading_srv, &prim_id_srv, &positions_srv, &indices_srv, &silhouette_counter_srv_,
                &silhouette_info_srv_, &grad_anti_aliased_srv};
            GpuUnorderedAccessView* uavs[] = {&grad_shading_uav, &grad_positions_uav};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.ComputeIndirect(anti_alias_bwd_pipeline_, indirect_args_, shader_binding);
        }
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
                GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, L"GpuDiffRender.GenerateMipmap.texture_mip");
            texture_mip = &texture_temp;

            if (texture.Format() != GpuFormat::RGBA32_Float)
            {
                constexpr uint32_t BlockDim = 16;

                GpuConstantBufferOfType<TextureCopyConstantBuffer> texture_copy_cb(gpu_system_, L"texture_copy_cb");
                texture_copy_cb->tex_size = glm::uvec2(width, height);
                texture_copy_cb.UploadStaging();

                const GpuShaderResourceView texture_srv(gpu_system_, texture, 0);

                GpuUnorderedAccessView texture_mip_uav(gpu_system_, *texture_mip, 0);

                const GpuConstantBuffer* cbs[] = {&texture_copy_cb};
                const GpuShaderResourceView* srvs[] = {&texture_srv};
                GpuUnorderedAccessView* uavs[] = {&texture_mip_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
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
        const GpuDynamicSampler& sampler, GpuTexture2D& image)
    {
        const uint32_t gbuffer_width = prim_id.Width(0);
        const uint32_t gbuffer_height = prim_id.Height(0);

        if ((image.Width(0) != gbuffer_width) || (image.Height(0) != gbuffer_height) || (image.Format() != texture.Format()))
        {
            image = GpuTexture2D(gpu_system_, gbuffer_width, gbuffer_height, 1, texture.Format(),
                GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        image.Name(L"GpuDiffRender.TextureFwd.image");

        constexpr uint32_t BlockDim = 16;

        GpuConstantBufferOfType<TextureFwdConstantBuffer> texture_fwd_cb(gpu_system_, L"texture_fwd_cb");
        texture_fwd_cb->gbuffer_size = glm::uvec2(gbuffer_width, gbuffer_height);
        texture_fwd_cb.UploadStaging();

        const GpuShaderResourceView prim_id_srv(gpu_system_, prim_id, GpuFormat::R32_Uint);
        const GpuShaderResourceView texture_srv(gpu_system_, texture);
        const GpuShaderResourceView uv_srv(gpu_system_, uv, GpuFormat::RG32_Float);

        GpuUnorderedAccessView image_uav(gpu_system_, image);

        const float clear_clr[] = {0, 0, 0, 0};
        cmd_list.Clear(image_uav, clear_clr);

        const GpuConstantBuffer* cbs[] = {&texture_fwd_cb};
        const GpuShaderResourceView* srvs[] = {&prim_id_srv, &texture_srv, &uv_srv};
        GpuUnorderedAccessView* uavs[] = {&image_uav};
        const GpuDynamicSampler* samplers[] = {&sampler};
        const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs, samplers};
        cmd_list.Compute(texture_fwd_pipeline_, DivUp(gbuffer_width, BlockDim), DivUp(gbuffer_height, BlockDim), 1, shader_binding);
    }

    void GpuDiffRender::TextureBwd(GpuCommandList& cmd_list, const GpuTexture2D& texture, const GpuTexture2D& prim_id, const GpuBuffer& uv,
        const GpuBuffer& grad_image, const GpuDynamicSampler& sampler, GpuBuffer& grad_texture, GpuBuffer& grad_uv)
    {
        const uint32_t gbuffer_width = prim_id.Width(0);
        const uint32_t gbuffer_height = prim_id.Height(0);
        const uint32_t tex_width = texture.Width(0);
        const uint32_t tex_height = texture.Height(0);

        const uint32_t grad_texture_size = tex_width * tex_height * FormatChannels(texture.Format()) * sizeof(float);
        if (grad_texture.Size() != grad_texture_size)
        {
            grad_texture =
                GpuBuffer(gpu_system_, grad_texture_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_texture.Name(L"GpuDiffRender.TextureBwd.grad_texture");

        const uint32_t grad_uv_size = gbuffer_width * gbuffer_height * sizeof(glm::uvec2);
        if (grad_uv.Size() != grad_uv_size)
        {
            grad_uv = GpuBuffer(gpu_system_, grad_uv_size, GpuHeap::Default, GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable);
        }
        grad_uv.Name(L"GpuDiffRender.TextureBwd.grad_uv");

        constexpr uint32_t BlockDim = 16;

        GpuConstantBufferOfType<TextureBwdConstantBuffer> texture_bwd_cb(gpu_system_, L"texture_bwd_cb");
        texture_bwd_cb->gbuffer_size = glm::uvec2(gbuffer_width, gbuffer_height);
        texture_bwd_cb->tex_size = glm::uvec2(tex_width, tex_height);
        texture_bwd_cb->num_channels = FormatChannels(texture.Format());
        texture_bwd_cb->min_mag_filter_linear = sampler.SamplerFilters().min == GpuSampler::Filter::Linear;
        texture_bwd_cb->address_mode = static_cast<uint32_t>(sampler.SamplerAddressModes().u);
        texture_bwd_cb.UploadStaging();

        const GpuShaderResourceView texture_srv(gpu_system_, texture);
        const GpuShaderResourceView uv_srv(gpu_system_, uv, GpuFormat::RG32_Float);
        const GpuShaderResourceView grad_image_srv(gpu_system_, grad_image, GpuFormat::R32_Float);

        GpuUnorderedAccessView grad_texture_uav(gpu_system_, grad_texture, GpuFormat::R32_Uint); // Float as uint due to atomic operations
        GpuUnorderedAccessView grad_uv_uav(gpu_system_, grad_uv, GpuFormat::RG32_Float);

        {
            const uint32_t clear_clr[] = {0, 0, 0, 0};
            cmd_list.Clear(grad_texture_uav, clear_clr);
        }
        {
            const float clear_clr[] = {0, 0, 0, 0};
            cmd_list.Clear(grad_uv_uav, clear_clr);
        }

        const GpuConstantBuffer* cbs[] = {&texture_bwd_cb};
        const GpuShaderResourceView* srvs[] = {&texture_srv, &uv_srv, &grad_image_srv};
        GpuUnorderedAccessView* uavs[] = {&grad_texture_uav, &grad_uv_uav};
        const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
        cmd_list.Compute(texture_bwd_pipeline_, DivUp(gbuffer_width, BlockDim), DivUp(gbuffer_height, BlockDim), 1, shader_binding);
    }
} // namespace AIHoloImager
