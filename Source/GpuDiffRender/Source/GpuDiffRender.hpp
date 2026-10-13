// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"

namespace AIHoloImager
{
    class GpuDiffRender
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDiffRender);

    public:
        GpuDiffRender() noexcept;
        explicit GpuDiffRender(GpuSystem& gpu_system);
        GpuDiffRender(GpuDiffRender&& other) noexcept;
        ~GpuDiffRender() noexcept;

        GpuDiffRender& operator=(GpuDiffRender&& other) noexcept;

        void RasterizeFwd(GpuCommandList& cmd_list, const GpuBuffer& positions, const GpuBuffer& indices, uint32_t width, uint32_t height,
            const GpuViewport& viewport, bool needs_derivative_barycentric, GpuTexture2D& barycentric,
            /*optional*/ GpuTexture2D& derivative_barycentric, GpuTexture2D& prim_id);
        void RasterizeBwd(GpuCommandList& cmd_list, const GpuBuffer& positions, const GpuBuffer& indices, const GpuViewport& viewport,
            const GpuTexture2D& barycentric, const GpuTexture2D& prim_id, const GpuTexture2D& grad_barycentric,
            /*optional*/ const GpuTexture2D& grad_derivative_barycentric, GpuBuffer& grad_positions);

        void InterpolateFwd(GpuCommandList& cmd_list, const GpuBuffer& vtx_attribs, uint32_t num_attribs_per_vtx,
            const GpuTexture2D& barycentric, /*optional*/ const GpuTexture2D& derivative_barycentric, const GpuTexture2D& prim_id,
            const GpuBuffer& indices, GpuBuffer& shading, /*optional*/ GpuBuffer& derivative_shading);
        void InterpolateBwd(GpuCommandList& cmd_list, const GpuBuffer& vtx_attribs, uint32_t num_attribs_per_vtx,
            const GpuTexture2D& barycentric, /*optional*/ const GpuTexture2D& derivative_barycentric, const GpuTexture2D& prim_id,
            const GpuBuffer& indices, const GpuBuffer& grad_shading, /*optional*/ const GpuBuffer& grad_derivative_shading,
            GpuBuffer& grad_vtx_attribs, GpuTexture2D& grad_barycentric, /*optional*/ GpuTexture2D& grad_derivative_barycentric);

        void GenerateMipmaps(GpuCommandList& cmd_list, GpuTexture2D& texture, uint32_t mip_levels);

        void TextureFwd(GpuCommandList& cmd_list, const GpuTexture2D& texture, const GpuTexture2D& prim_id, const GpuBuffer& uv,
            /*optional*/ const GpuBuffer& derivative_uv, const GpuDynamicSampler& sampler, GpuTexture2D& image);
        void TextureBwd(GpuCommandList& cmd_list, const GpuTexture2D& texture, const GpuTexture2D& prim_id, const GpuBuffer& uv,
            /*optional*/ const GpuBuffer& derivative_uv, const GpuTexture2D& grad_image, const GpuDynamicSampler& sampler,
            GpuBuffer& grad_texture, GpuBuffer& grad_uv, /*optional*/ GpuBuffer& grad_derivative_uv);

        void AntiAliasConstructOppositeVertices(GpuCommandList& cmd_list, const GpuBuffer& indices, GpuBuffer& opposite_vertices);

        void AntiAliasFwd(GpuCommandList& cmd_list, const GpuBuffer& shading, const GpuTexture2D& prim_id, const GpuBuffer& positions,
            const GpuBuffer& indices, const GpuViewport& viewport, const GpuBuffer& opposite_vertices, GpuBuffer& anti_aliased);
        void AntiAliasBwd(GpuCommandList& cmd_list, const GpuBuffer& shading, const GpuTexture2D& prim_id, const GpuBuffer& positions,
            const GpuBuffer& indices, const GpuViewport& viewport, const GpuBuffer& grad_anti_aliased, GpuBuffer& grad_shading,
            GpuBuffer& grad_positions);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
