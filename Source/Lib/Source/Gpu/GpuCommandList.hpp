// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <span>

#include "GpuBufferHelper.hpp"
#include "GpuResourceViews.hpp"
#include "GpuShader.hpp"
#include "GpuSystem.hpp"
#include "GpuTexture.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    struct GpuViewport
    {
        float left;
        float top;
        float width;
        float height;
        float min_depth = 0;
        float max_depth = 1;
    };

    struct GpuRect
    {
        int32_t left;
        int32_t top;
        int32_t right;
        int32_t bottom;
    };

    class GpuCommandList
    {
        DISALLOW_COPY_AND_ASSIGN(GpuCommandList)

    public:
        struct VertexBufferBinding
        {
            const GpuBuffer* vb;
            uint32_t offset;
            uint32_t stride;
        };

        struct IndexBufferBinding
        {
            const GpuBuffer* ib;
            uint32_t offset;
            DXGI_FORMAT format;
        };

        struct ShaderBinding
        {
            std::span<const GeneralConstantBuffer*> cbs;
            std::span<const GpuShaderResourceView*> srvs;
            std::span<GpuUnorderedAccessView*> uavs;
        };

    public:
        GpuCommandList() noexcept;
        GpuCommandList(GpuSystem& gpu_system, ID3D12CommandAllocator* cmd_allocator, GpuSystem::CmdQueueType type);
        ~GpuCommandList();

        GpuCommandList(GpuCommandList&& other) noexcept;
        GpuCommandList& operator=(GpuCommandList&& other) noexcept;

        GpuSystem::CmdQueueType Type() const noexcept;

        explicit operator bool() const noexcept;

        ID3D12CommandList* NativeCommandListBase() const noexcept
        {
            return cmd_list_.Get();
        }
        template <typename T>
        T* NativeCommandList() const;

        void Transition(std::span<const D3D12_RESOURCE_BARRIER> barriers) const noexcept;

        void Clear(GpuRenderTargetView& rtv, const float color[4]);
        void Clear(GpuUnorderedAccessView& uav, const float color[4]);
        void Clear(GpuUnorderedAccessView& uav, const uint32_t color[4]);
        void ClearDepth(GpuDepthStencilView& dsv, float depth);
        void ClearStencil(GpuDepthStencilView& dsv, uint8_t stencil);
        void ClearDepthStencil(GpuDepthStencilView& dsv, float depth, uint8_t stencil);

        void Render(const GpuRenderPipeline& pipeline, std::span<const VertexBufferBinding> vbs, const IndexBufferBinding* ib, uint32_t num,
            const ShaderBinding shader_bindings[GpuRenderPipeline::NumShaderStages], std::span<const GpuRenderTargetView*> rtvs,
            const GpuDepthStencilView* dsv, std::span<const GpuViewport> viewports, std::span<const GpuRect> scissor_rects);
        void Compute(
            const GpuComputePipeline& pipeline, uint32_t group_x, uint32_t group_y, uint32_t group_z, const ShaderBinding& shader_binding);
        void Copy(GpuBuffer& dest, const GpuBuffer& src);
        void Copy(GpuBuffer& dest, uint32_t dst_offset, const GpuBuffer& src, uint32_t src_offset, uint32_t src_size);
        void Copy(GpuTexture2D& dest, const GpuTexture2D& src);

        void Close();
        void Reset(ID3D12CommandAllocator* cmd_allocator);

    private:
        GpuSystem* gpu_system_ = nullptr;

        GpuSystem::CmdQueueType type_ = GpuSystem::CmdQueueType::Num;
        ComPtr<ID3D12CommandList> cmd_list_;
        bool closed_ = false;
    };
} // namespace AIHoloImager
