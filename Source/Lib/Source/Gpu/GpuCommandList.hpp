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
            std::span<const GpuTexture2D*> srvs;
            std::span<GpuTexture2D*> uavs;
        };

        struct RenderTargetBinding
        {
            GpuTexture2D* texture;
            GpuRenderTargetView* rtv;
        };

        struct DepthStencilBinding
        {
            GpuTexture2D* texture;
            GpuDepthStencilView* dsv;
        };

    public:
        GpuCommandList() noexcept;
        GpuCommandList(GpuSystem& gpu_system, ID3D12CommandAllocator* cmd_allocator, GpuSystem::CmdQueueType type);
        ~GpuCommandList() noexcept;

        GpuCommandList(GpuCommandList&& other) noexcept;
        GpuCommandList& operator=(GpuCommandList&& other) noexcept;

        GpuSystem::CmdQueueType Type() const noexcept;

        explicit operator bool() const noexcept;

        ID3D12CommandList* NativeCommandListBase() const noexcept
        {
            return cmd_list_.Get();
        }
        template <typename T>
        T* NativeCommandList() const noexcept
        {
            return static_cast<T*>(NativeCommandListBase());
        }

        void Transition(std::span<const D3D12_RESOURCE_BARRIER> barriers) const noexcept;

        void Render(const GpuRenderPipeline& pipeline, std::span<const VertexBufferBinding> vbs, const IndexBufferBinding* ib, uint32_t num,
            const ShaderBinding shader_bindings[GpuRenderPipeline::NumShaderStages], std::span<const RenderTargetBinding> rts,
            const DepthStencilBinding* ds, std::span<const D3D12_VIEWPORT> viewports, std::span<const D3D12_RECT> scissor_rects);
        void Compute(
            const GpuComputePipeline& pipeline, uint32_t group_x, uint32_t group_y, uint32_t group_z, const ShaderBinding& shader_binding);

        void Close();
        void Reset(ID3D12CommandAllocator* cmd_allocator);

    private:
        GpuSystem* gpu_system_ = nullptr;

        GpuSystem::CmdQueueType type_ = GpuSystem::CmdQueueType::Num;
        ComPtr<ID3D12CommandList> cmd_list_;
    };
} // namespace AIHoloImager
