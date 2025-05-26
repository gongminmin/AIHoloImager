// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <future>
#include <span>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuConstantBuffer.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"

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

    struct GpuBox
    {
        uint32_t left;
        uint32_t top;
        uint32_t front;
        uint32_t right;
        uint32_t bottom;
        uint32_t back;
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
            GpuFormat format;
        };

        struct ShaderBinding
        {
            std::span<const GpuConstantBuffer*> cbs;
            std::span<const GpuShaderResourceView*> srvs;
            std::span<GpuUnorderedAccessView*> uavs;
        };

    public:
        GpuCommandList() noexcept;
        GpuCommandList(GpuSystem& gpu_system, GpuCommandAllocatorInfo& cmd_alloc_info, GpuSystem::CmdQueueType type);
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
            std::span<const ShaderBinding> shader_bindings, std::span<const GpuRenderTargetView*> rtvs, const GpuDepthStencilView* dsv,
            std::span<const GpuViewport> viewports, std::span<const GpuRect> scissor_rects);
        void Compute(
            const GpuComputePipeline& pipeline, uint32_t group_x, uint32_t group_y, uint32_t group_z, const ShaderBinding& shader_binding);
        void ComputeIndirect(const GpuComputePipeline& pipeline, const GpuBuffer& indirect_args, const ShaderBinding& shader_binding);
        void Copy(GpuBuffer& dest, const GpuBuffer& src);
        void Copy(GpuBuffer& dest, uint32_t dst_offset, const GpuBuffer& src, uint32_t src_offset, uint32_t src_size);
        void Copy(GpuTexture& dest, const GpuTexture& src);
        void Copy(GpuTexture& dest, uint32_t dest_sub_resource, uint32_t dst_x, uint32_t dst_y, uint32_t dst_z, const GpuTexture& src,
            uint32_t src_sub_resource, const GpuBox& src_box);

        void Upload(GpuBuffer& dest, const std::function<void(void*)>& copy_func);
        void Upload(GpuBuffer& dest, const void* src_data, uint32_t src_size);
        void Upload(
            GpuTexture& dest, uint32_t sub_resource, const std::function<void(void*, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func);
        void Upload(GpuTexture& dest, uint32_t sub_resource, const void* src_data, uint32_t src_size);
        [[nodiscard]] std::future<void> ReadBackAsync(const GpuBuffer& src, const std::function<void(const void*)>& copy_func);
        [[nodiscard]] std::future<void> ReadBackAsync(const GpuBuffer& src, void* dst_data, uint32_t dst_size);
        [[nodiscard]] std::future<void> ReadBackAsync(const GpuTexture& src, uint32_t sub_resource,
            const std::function<void(const void*, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func);
        [[nodiscard]] std::future<void> ReadBackAsync(const GpuTexture& src, uint32_t sub_resource, void* dst_data, uint32_t dst_size);

        void Close();
        void Reset(GpuCommandAllocatorInfo& cmd_alloc_info);

        GpuCommandAllocatorInfo* CommandAllocatorInfo() noexcept
        {
            return cmd_alloc_info_;
        }

    private:
        GpuDescriptorBlock BindPipeline(const GpuComputePipeline& pipeline, const ShaderBinding& shader_binding);

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuCommandAllocatorInfo* cmd_alloc_info_ = nullptr;

        GpuSystem::CmdQueueType type_ = GpuSystem::CmdQueueType::Num;
        ComPtr<ID3D12CommandList> cmd_list_;
        bool closed_ = false;
    };
} // namespace AIHoloImager
