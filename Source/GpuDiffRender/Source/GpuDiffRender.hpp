// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"

namespace AIHoloImager
{
    class GpuDiffRender
    {
    public:
        explicit GpuDiffRender(GpuSystem& gpu_system);
        ~GpuDiffRender();

        void RasterizeFwd(GpuCommandList& cmd_list, const GpuBuffer& positions, const GpuBuffer& indices, uint32_t width, uint32_t height,
            const GpuViewport& viewport, GpuTexture2D& barycentric, GpuTexture2D& prim_id);
        void RasterizeBwd(GpuCommandList& cmd_list, const GpuBuffer& positions, const GpuBuffer& indices, const GpuViewport& viewport,
            const GpuTexture2D& barycentric, const GpuTexture2D& prim_id, const GpuTexture2D& grad_barycentric, GpuBuffer& grad_positions);

        void InterpolateFwd(GpuCommandList& cmd_list, const GpuBuffer& vtx_attribs, uint32_t num_attribs_per_vtx,
            const GpuTexture2D& barycentric, const GpuTexture2D& prim_id, const GpuBuffer& indices, GpuBuffer& shading);
        void InterpolateBwd(GpuCommandList& cmd_list, const GpuBuffer& vtx_attribs, uint32_t num_attribs_per_vtx,
            const GpuTexture2D& barycentric, const GpuTexture2D& prim_id, const GpuBuffer& indices, const GpuBuffer& grad_shading,
            GpuBuffer& grad_vtx_attribs, GpuTexture2D& grad_barycentric);

        void AntiAliasConstructOppositeVertices(GpuCommandList& cmd_list, const GpuBuffer& indices, GpuBuffer& opposite_vertices);

        void AntiAliasFwd(GpuCommandList& cmd_list, const GpuBuffer& shading, const GpuTexture2D& prim_id, const GpuBuffer& positions,
            const GpuBuffer& indices, const GpuViewport& viewport, const GpuBuffer& opposite_vertices, GpuBuffer& anti_aliased);
        void AntiAliasBwd(GpuCommandList& cmd_list, const GpuBuffer& shading, const GpuTexture2D& prim_id, const GpuBuffer& positions,
            const GpuBuffer& indices, const GpuViewport& viewport, const GpuBuffer& grad_anti_aliased, GpuBuffer& grad_shading,
            GpuBuffer& grad_positions);

    private:
        GpuSystem& gpu_system_;

        GpuTexture2D depth_tex_;
        GpuShaderResourceView depth_srv_;
        GpuDepthStencilView depth_dsv_;
        GpuRenderPipeline rasterize_fwd_pipeline_;

        struct RasterizeBwdConstantBuffer
        {
            glm::vec4 viewport;
            glm::uvec2 gbuffer_size;
            uint32_t padding[2];
        };
        GpuConstantBufferOfType<RasterizeBwdConstantBuffer> rasterize_bwd_cb_;
        GpuComputePipeline rasterize_bwd_pipeline_;

        struct InterpolateFwdConstantBuffer
        {
            glm::uvec2 gbuffer_size;
            uint32_t num_attribs;
            uint32_t padding[1];
        };
        GpuConstantBufferOfType<InterpolateFwdConstantBuffer> interpolate_fwd_cb_;
        GpuComputePipeline interpolate_fwd_pipeline_;

        struct InterpolateBwdConstantBuffer
        {
            glm::uvec2 gbuffer_size;
            uint32_t num_attribs;
            uint32_t padding[1];
        };
        GpuConstantBufferOfType<InterpolateBwdConstantBuffer> interpolate_bwd_cb_;
        GpuComputePipeline interpolate_bwd_pipeline_;

        GpuComputePipeline anti_alias_indirect_pipeline_;

        GpuBuffer silhouette_counter_;
        GpuShaderResourceView silhouette_counter_srv_;
        GpuUnorderedAccessView silhouette_counter_uav_;
        GpuBuffer silhouette_info_;
        GpuShaderResourceView silhouette_info_srv_;
        GpuUnorderedAccessView silhouette_info_uav_;
        GpuBuffer opposite_vertices_hash_;
        GpuShaderResourceView opposite_vertices_hash_srv_;
        GpuUnorderedAccessView opposite_vertices_hash_uav_;

        struct AntiAliasConstructOppositeVerticesHashConstantBuffer
        {
            uint32_t num_indices;
            uint32_t hash_size;
            uint32_t padding[2];
        };
        GpuConstantBufferOfType<AntiAliasConstructOppositeVerticesHashConstantBuffer> anti_alias_construct_oppo_vert_hash_cb_;
        GpuComputePipeline anti_alias_construct_oppo_vert_hash_pipeline_;

        struct AntiAliasConstructOppositeVerticesConstantBuffer
        {
            uint32_t num_indices;
            uint32_t hash_size;
            uint32_t padding[2];
        };
        GpuConstantBufferOfType<AntiAliasConstructOppositeVerticesConstantBuffer> anti_alias_construct_oppo_vert_cb_;
        GpuComputePipeline anti_alias_construct_oppo_vert_pipeline_;

        struct AntiAliasFwdConstantBuffer
        {
            glm::vec4 viewport;
            glm::uvec2 gbuffer_size;
            uint32_t num_attribs;
            uint32_t padding[1];
        };
        GpuConstantBufferOfType<AntiAliasFwdConstantBuffer> anti_alias_fwd_cb_;
        GpuComputePipeline anti_alias_fwd_pipeline_;

        struct AntiAliasIndirectArgsConstantBuffer
        {
            uint32_t bwd_block_dim;
            uint32_t padding[3];
        };
        GpuConstantBufferOfType<AntiAliasIndirectArgsConstantBuffer> anti_alias_indirect_args_cb_;
        GpuBuffer indirect_args_;
        GpuUnorderedAccessView indirect_args_uav_;

        struct AntiAliasBwdConstantBuffer
        {
            glm::vec4 viewport;
            glm::uvec2 gbuffer_size;
            uint32_t num_attribs;
            uint32_t padding[1];
        };
        GpuConstantBufferOfType<AntiAliasBwdConstantBuffer> anti_alias_bwd_cb_;
        GpuComputePipeline anti_alias_bwd_pipeline_;
    };
} // namespace AIHoloImager
