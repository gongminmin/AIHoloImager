// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuSystem.hpp"

#include "GpuBufferInternal.hpp"
#include "GpuCommandListInternal.hpp"
#include "GpuCommandPoolInternal.hpp"
#include "GpuResourceViewsInternal.hpp"
#include "GpuSamplerInternal.hpp"
#include "GpuShaderInternal.hpp"
#include "GpuTextureInternal.hpp"
#include "GpuVertexLayoutInternal.hpp"

namespace AIHoloImager
{
    class GpuSystemInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuSystemInternal)

    public:
        GpuSystemInternal() noexcept;
        virtual ~GpuSystemInternal();

        GpuSystemInternal(GpuSystemInternal&& other) noexcept;
        virtual GpuSystemInternal& operator=(GpuSystemInternal&& other) noexcept = 0;

        virtual void* NativeDevice() const noexcept = 0;
        virtual void* NativeCommandQueue(GpuSystem::CmdQueueType type) const noexcept = 0;

        virtual LUID DeviceLuid() const = 0;

        virtual void* SharedFenceHandle(GpuSystem::CmdQueueType type) const noexcept = 0;

        virtual [[nodiscard]] GpuCommandList CreateCommandList(GpuSystem::CmdQueueType type) = 0;
        virtual uint64_t Execute(GpuCommandList&& cmd_list, GpuSystem::CmdQueueType wait_queue_type, uint64_t wait_fence_value) = 0;
        virtual uint64_t ExecuteAndReset(GpuCommandList& cmd_list, GpuSystem::CmdQueueType wait_queue_type, uint64_t wait_fence_value) = 0;

        virtual uint32_t ConstantDataAlignment() const noexcept = 0;
        virtual uint32_t StructuredDataAlignment() const noexcept = 0;
        virtual uint32_t TextureDataAlignment() const noexcept = 0;

        virtual void CpuWait(GpuSystem::CmdQueueType wait_queue_type, uint64_t wait_fence_value) = 0;
        virtual void GpuWait(
            GpuSystem::CmdQueueType target_queue_type, GpuSystem::CmdQueueType wait_queue_type, uint64_t wait_fence_value) = 0;
        virtual uint64_t FenceValue(GpuSystem::CmdQueueType type) const noexcept = 0;
        virtual uint64_t CompletedFenceValue(GpuSystem::CmdQueueType type) const = 0;

        virtual void HandleDeviceLost() = 0;
        virtual void ClearStallResources() = 0;

        virtual std::unique_ptr<GpuBufferInternal> CreateBuffer(
            uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name) const = 0;

        virtual std::unique_ptr<GpuTextureInternal> CreateTexture(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
            uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::string_view name) const = 0;

        virtual std::unique_ptr<GpuStaticSamplerInternal> CreateStaticSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const = 0;
        virtual std::unique_ptr<GpuDynamicSamplerInternal> CreateDynamicSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const = 0;

        virtual std::unique_ptr<GpuVertexLayoutInternal> CreateVertexLayout(
            std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides) const = 0;

        virtual std::unique_ptr<GpuConstantBufferViewInternal> CreateConstantBufferView(
            const GpuBuffer& buffer, uint32_t offset, uint32_t size) const = 0;

        virtual std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const = 0;
        virtual std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const = 0;
        virtual std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const = 0;
        virtual std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const = 0;
        virtual std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const = 0;

        virtual std::unique_ptr<GpuRenderTargetViewInternal> CreateRenderTargetView(GpuTexture2D& texture, GpuFormat format) const = 0;

        virtual std::unique_ptr<GpuDepthStencilViewInternal> CreateDepthStencilView(GpuTexture2D& texture, GpuFormat format) const = 0;

        virtual std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const = 0;
        virtual std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const = 0;
        virtual std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const = 0;
        virtual std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const = 0;
        virtual std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const = 0;

        virtual std::unique_ptr<GpuRenderPipelineInternal> CreateRenderPipeline(GpuRenderPipeline::PrimitiveTopology topology,
            std::span<const ShaderInfo> shaders, const GpuVertexLayout& vertex_layout, std::span<const GpuStaticSampler> static_samplers,
            const GpuRenderPipeline::States& states) const = 0;
        virtual std::unique_ptr<GpuComputePipelineInternal> CreateComputePipeline(
            const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers) const = 0;

        virtual std::unique_ptr<GpuCommandPoolInternal> CreateCommandPool(GpuSystem::CmdQueueType type) const = 0;

        virtual std::unique_ptr<GpuCommandListInternal> CreateCommandList(GpuCommandPool& cmd_pool, GpuSystem::CmdQueueType type) const = 0;
    };
} // namespace AIHoloImager
