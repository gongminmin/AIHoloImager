// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuCommandList.hpp"

namespace AIHoloImager
{
    class GpuCommandListInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuCommandListInternal)

    public:
        GpuCommandListInternal() noexcept;
        virtual ~GpuCommandListInternal();

        GpuCommandListInternal(GpuCommandListInternal&& other) noexcept;
        virtual GpuCommandListInternal& operator=(GpuCommandListInternal&& other) noexcept = 0;

        virtual GpuSystem::CmdQueueType Type() const noexcept = 0;

        virtual explicit operator bool() const noexcept = 0;

        virtual void* NativeCommandListBase() const noexcept = 0;

        virtual void Clear(GpuRenderTargetView& rtv, const float color[4]) = 0;
        virtual void Clear(GpuUnorderedAccessView& uav, const float color[4]) = 0;
        virtual void Clear(GpuUnorderedAccessView& uav, const uint32_t color[4]) = 0;
        virtual void ClearDepth(GpuDepthStencilView& dsv, float depth) = 0;
        virtual void ClearStencil(GpuDepthStencilView& dsv, uint8_t stencil) = 0;
        virtual void ClearDepthStencil(GpuDepthStencilView& dsv, float depth, uint8_t stencil) = 0;

        virtual void Render(const GpuRenderPipeline& pipeline, std::span<const GpuCommandList::VertexBufferBinding> vbs,
            const GpuCommandList::IndexBufferBinding* ib, uint32_t num, std::span<const GpuCommandList::ShaderBinding> shader_bindings,
            std::span<const GpuRenderTargetView*> rtvs, const GpuDepthStencilView* dsv, std::span<const GpuViewport> viewports,
            std::span<const GpuRect> scissor_rects) = 0;
        virtual void Compute(const GpuComputePipeline& pipeline, uint32_t group_x, uint32_t group_y, uint32_t group_z,
            const GpuCommandList::ShaderBinding& shader_binding) = 0;
        virtual void ComputeIndirect(
            const GpuComputePipeline& pipeline, const GpuBuffer& indirect_args, const GpuCommandList::ShaderBinding& shader_binding) = 0;
        virtual void Copy(GpuBuffer& dest, const GpuBuffer& src) = 0;
        virtual void Copy(GpuBuffer& dest, uint32_t dst_offset, const GpuBuffer& src, uint32_t src_offset, uint32_t src_size) = 0;
        virtual void Copy(GpuTexture& dest, const GpuTexture& src) = 0;
        virtual void Copy(GpuTexture& dest, uint32_t dest_sub_resource, uint32_t dst_x, uint32_t dst_y, uint32_t dst_z,
            const GpuTexture& src, uint32_t src_sub_resource, const GpuBox& src_box) = 0;

        virtual void Upload(GpuBuffer& dest, const std::function<void(void* dst_data)>& copy_func) = 0;
        virtual void Upload(GpuTexture& dest, uint32_t sub_resource,
            const std::function<void(void* dst_data, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func) = 0;
        virtual [[nodiscard]] std::future<void> ReadBackAsync(
            const GpuBuffer& src, const std::function<void(const void* src_data)>& copy_func) = 0;
        virtual [[nodiscard]] std::future<void> ReadBackAsync(const GpuTexture& src, uint32_t sub_resource,
            const std::function<void(const void* src_data, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func) = 0;

        virtual void Close() = 0;
        virtual void Reset(GpuCommandPool& cmd_pool) = 0;
    };
} // namespace AIHoloImager
