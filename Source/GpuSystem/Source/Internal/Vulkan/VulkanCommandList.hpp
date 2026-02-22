// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include <volk.h>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandListInternal.hpp"
#include "VulkanImpDefine.hpp"

namespace AIHoloImager
{
    struct VulkanBindingSlots;

    class VulkanCommandList : public GpuCommandListInternal
    {
    public:
        VulkanCommandList(GpuSystem& gpu_system, GpuCommandPool& cmd_pool, GpuSystem::CmdQueueType type);
        ~VulkanCommandList() override;

        VulkanCommandList(VulkanCommandList&& other) noexcept;
        explicit VulkanCommandList(GpuCommandListInternal&& other) noexcept;
        VulkanCommandList& operator=(VulkanCommandList&& other) noexcept;
        GpuCommandListInternal& operator=(GpuCommandListInternal&& other) noexcept override;

        GpuSystem::CmdQueueType Type() const noexcept override;

        explicit operator bool() const noexcept override;

        VkCommandBuffer CommandBuffer() const noexcept
        {
            return cmd_buff_;
        }
        void* NativeCommandListBase() const noexcept override
        {
            return this->CommandBuffer();
        }

        void Transition(std::span<const VkBufferMemoryBarrier> barriers) const noexcept;
        void Transition(std::span<const VkImageMemoryBarrier> barriers) const noexcept;

        void Clear(GpuRenderTargetView& rtv, const float color[4]) override;
        void Clear(GpuUnorderedAccessView& uav, const float color[4]) override;
        void Clear(GpuUnorderedAccessView& uav, const uint32_t color[4]) override;
        void ClearDepth(GpuDepthStencilView& dsv, float depth) override;
        void ClearStencil(GpuDepthStencilView& dsv, uint8_t stencil) override;
        void ClearDepthStencil(GpuDepthStencilView& dsv, float depth, uint8_t stencil) override;

        void Render(const GpuRenderPipeline& pipeline, std::span<const GpuCommandList::VertexBufferBinding> vbs,
            const GpuCommandList::IndexBufferBinding* ib, uint32_t num, std::span<const GpuCommandList::ShaderBinding> shader_bindings,
            std::span<GpuRenderTargetView*> rtvs, GpuDepthStencilView* dsv, std::span<const GpuViewport> viewports,
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
        void Reset(GpuCommandPool& cmd_pool) override;

        GpuCommandPool* CommandPool() noexcept
        {
            return cmd_pool_;
        }

    private:
        void Compute(
            const GpuComputePipeline& pipeline, const GpuCommandList::ShaderBinding& shader_binding, std::function<void()> dispatch_call);
        void GenWriteDescSet(std::vector<VkWriteDescriptorSet>& write_desc_sets, const VulkanBindingSlots& binding_slots,
            std::string_view shader_name, const GpuCommandList::ShaderBinding& shader_binding, std::span<const VkDescriptorSet> desc_sets);

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuCommandPool* cmd_pool_ = nullptr;

        GpuSystem::CmdQueueType type_ = GpuSystem::CmdQueueType::Num;
        VkCommandBuffer cmd_buff_;
        bool closed_ = false;
    };

    VULKAN_DEFINE_IMP(CommandList)
} // namespace AIHoloImager
