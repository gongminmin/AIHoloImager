// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandListInternal.hpp"
#include "D3D12ImpDefine.hpp"

namespace AIHoloImager
{
    class D3D12CommandList : public GpuCommandListInternal
    {
    public:
        D3D12CommandList(GpuSystem& gpu_system, GpuCommandAllocatorInfo& cmd_alloc_info, GpuSystem::CmdQueueType type);
        ~D3D12CommandList() override;

        D3D12CommandList(D3D12CommandList&& other) noexcept;
        explicit D3D12CommandList(GpuCommandListInternal&& other) noexcept;
        D3D12CommandList& operator=(D3D12CommandList&& other) noexcept;
        GpuCommandListInternal& operator=(GpuCommandListInternal&& other) noexcept override;

        GpuSystem::CmdQueueType Type() const noexcept override;

        explicit operator bool() const noexcept override;

        ID3D12CommandList* CommandListBase() const noexcept
        {
            return cmd_list_.Get();
        }
        void* NativeCommandListBase() const noexcept override
        {
            return this->CommandListBase();
        }
        template <typename Traits>
        typename Traits::CommandListType NativeCommandListBase() const noexcept
        {
            return reinterpret_cast<typename Traits::CommandListType>(this->NativeCommandListBase());
        }
        template <typename T>
        T* NativeCommandList() const;

        void Transition(std::span<const D3D12_RESOURCE_BARRIER> barriers) const noexcept;

        void Clear(GpuRenderTargetView& rtv, const float color[4]) override;
        void Clear(GpuUnorderedAccessView& uav, const float color[4]) override;
        void Clear(GpuUnorderedAccessView& uav, const uint32_t color[4]) override;
        void ClearDepth(GpuDepthStencilView& dsv, float depth) override;
        void ClearStencil(GpuDepthStencilView& dsv, uint8_t stencil) override;
        void ClearDepthStencil(GpuDepthStencilView& dsv, float depth, uint8_t stencil) override;

        void Render(const GpuRenderPipeline& pipeline, std::span<const GpuCommandList::VertexBufferBinding> vbs,
            const GpuCommandList::IndexBufferBinding* ib, uint32_t num, std::span<const GpuCommandList::ShaderBinding> shader_bindings,
            std::span<const GpuRenderTargetView*> rtvs, const GpuDepthStencilView* dsv, std::span<const GpuViewport> viewports,
            std::span<const GpuRect> scissor_rects) override;
        void Compute(const GpuComputePipeline& pipeline, uint32_t group_x, uint32_t group_y, uint32_t group_z,
            const GpuCommandList::ShaderBinding& shader_binding) override;
        void ComputeIndirect(const GpuComputePipeline& pipeline, const GpuBuffer& indirect_args,
            const GpuCommandList::ShaderBinding& shader_binding) override;
        void Copy(GpuBuffer& dest, const GpuBuffer& src) override;
        void Copy(GpuBuffer& dest, uint32_t dst_offset, const GpuBuffer& src, uint32_t src_offset, uint32_t src_size) override;
        void Copy(GpuTexture& dest, const GpuTexture& src) override;
        void Copy(GpuTexture& dest, uint32_t dest_sub_resource, uint32_t dst_x, uint32_t dst_y, uint32_t dst_z, const GpuTexture& src,
            uint32_t src_sub_resource, const GpuBox& src_box) override;

        void Upload(GpuBuffer& dest, const std::function<void(void* dst_data)>& copy_func) override;
        void Upload(GpuTexture& dest, uint32_t sub_resource,
            const std::function<void(void* dst_data, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func) override;
        [[nodiscard]] std::future<void> ReadBackAsync(
            const GpuBuffer& src, const std::function<void(const void* src_data)>& copy_func) override;
        [[nodiscard]] std::future<void> ReadBackAsync(const GpuTexture& src, uint32_t sub_resource,
            const std::function<void(const void* src_data, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func) override;

        void Close() override;
        void Reset(GpuCommandAllocatorInfo& cmd_alloc_info) override;

        GpuCommandAllocatorInfo* CommandAllocatorInfo() noexcept override
        {
            return cmd_alloc_info_;
        }

    private:
        GpuDescriptorBlock BindPipeline(const GpuComputePipeline& pipeline, const GpuCommandList::ShaderBinding& shader_binding);

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuCommandAllocatorInfo* cmd_alloc_info_ = nullptr;

        GpuSystem::CmdQueueType type_ = GpuSystem::CmdQueueType::Num;
        ComPtr<ID3D12CommandList> cmd_list_;
        bool closed_ = false;
    };

    D3D12_DEFINE_IMP(CommandList)
} // namespace AIHoloImager
