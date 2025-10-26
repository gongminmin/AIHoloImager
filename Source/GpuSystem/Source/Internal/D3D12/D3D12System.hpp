// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <functional>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandListInternal.hpp"
#include "../GpuSystemInternal.hpp"
#include "D3D12CommandList.hpp"

namespace AIHoloImager
{
    class D3D12System : public GpuSystemInternal
    {
    public:
        D3D12System(GpuSystem& gpu_system_, std::function<bool(void* device)> confirm_device = nullptr, bool enable_sharing = false,
            bool enable_debug = false);
        ~D3D12System() override;

        D3D12System(D3D12System&& other) noexcept;
        explicit D3D12System(GpuSystemInternal&& other) noexcept;
        D3D12System& operator=(D3D12System&& other) noexcept;
        GpuSystemInternal& operator=(GpuSystemInternal&& other) noexcept override;

        ID3D12Device* Device() const noexcept;
        void* NativeDevice() const noexcept override;
        template <typename Traits>
        typename Traits::DeviceType NativeDevice() const noexcept
        {
            return reinterpret_cast<typename Traits::DeviceType>(this->NativeDevice());
        }
        ID3D12CommandQueue* CommandQueue(GpuSystem::CmdQueueType type) const noexcept;
        void* NativeCommandQueue(GpuSystem::CmdQueueType type) const noexcept override;
        template <typename Traits>
        typename Traits::CommandQueueType NativeCommandQueue() const noexcept
        {
            return reinterpret_cast<typename Traits::CommandQueueType>(this->NativeCommandQueue());
        }

        void* SharedFenceHandle() const noexcept override;

        [[nodiscard]] GpuCommandList CreateCommandList(GpuSystem::CmdQueueType type) override;
        uint64_t Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value) override;
        uint64_t ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value) override;
        uint64_t ExecuteAndReset(D3D12CommandList& cmd_list, uint64_t wait_fence_value);

        uint32_t ConstantDataAlignment() const noexcept override;
        uint32_t StructuredDataAlignment() const noexcept override;
        uint32_t TextureDataAlignment() const noexcept override;

        void CpuWait(uint64_t fence_value) override;
        void GpuWait(GpuSystem::CmdQueueType type, uint64_t fence_value) override;
        uint64_t FenceValue() const noexcept override;
        uint64_t CompletedFenceValue() const override;

        void HandleDeviceLost() override;
        void ClearStallResources() override;

        void Recycle(ComPtr<ID3D12DeviceChild>&& resource);

        ID3D12CommandSignature* NativeDispatchIndirectSignature() const noexcept;

        std::unique_ptr<GpuBufferInternal> CreateBuffer(
            uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name) const override;
        std::unique_ptr<GpuBufferInternal> CreateBuffer(
            void* native_resource, GpuResourceState curr_state, std::wstring_view name) const override;

        std::unique_ptr<GpuTextureInternal> CreateTexture(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
            uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::wstring_view name) const override;
        std::unique_ptr<GpuTextureInternal> CreateTexture(
            void* native_resource, GpuResourceState curr_state, std::wstring_view name) const override;

        std::unique_ptr<GpuStaticSamplerInternal> CreateStaticSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const override;
        std::unique_ptr<GpuDynamicSamplerInternal> CreateDynamicSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const override;

        std::unique_ptr<GpuVertexAttribsInternal> CreateVertexAttribs(std::span<const GpuVertexAttrib> attribs) const override;

        std::unique_ptr<GpuDescriptorHeapInternal> CreateDescriptorHeap(
            uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name) const override;

        uint32_t DescriptorSize(GpuDescriptorHeapType type) const override;

        std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const override;
        std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const override;

        std::unique_ptr<GpuRenderTargetViewInternal> CreateRenderTargetView(GpuTexture2D& texture, GpuFormat format) const override;

        std::unique_ptr<GpuDepthStencilViewInternal> CreateDepthStencilView(GpuTexture2D& texture, GpuFormat format) const override;

        std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const override;
        std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const override;

        std::unique_ptr<GpuRenderPipelineInternal> CreateRenderPipeline(GpuRenderPipeline::PrimitiveTopology topology,
            std::span<const ShaderInfo> shaders, const GpuVertexAttribs& vertex_attribs, std::span<const GpuStaticSampler> static_samplers,
            const GpuRenderPipeline::States& states) const override;
        std::unique_ptr<GpuComputePipelineInternal> CreateComputePipeline(
            const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers) const override;

        std::unique_ptr<GpuCommandAllocatorInfoInternal> CreateCommandAllocatorInfo(GpuSystem::CmdQueueType type) const override;

        std::unique_ptr<GpuCommandListInternal> CreateCommandList(
            GpuCommandAllocatorInfo& cmd_alloc_info, GpuSystem::CmdQueueType type) const override;

    private:
        struct CmdQueue
        {
            ComPtr<ID3D12CommandQueue> cmd_queue;
            std::vector<std::unique_ptr<GpuCommandAllocatorInfo>> cmd_allocator_infos;
            std::list<GpuCommandList> free_cmd_lists;
        };

    private:
        CmdQueue& GetOrCreateCommandQueue(GpuSystem::CmdQueueType type);
        GpuCommandAllocatorInfo& CurrentCommandAllocator(GpuSystem::CmdQueueType type);
        uint64_t ExecuteOnly(D3D12CommandList& cmd_list, uint64_t wait_fence_value);

    private:
        GpuSystem* gpu_system_ = nullptr;

        ComPtr<ID3D12Device> device_;

        CmdQueue cmd_queues_[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)];

        ComPtr<ID3D12Fence> fence_;
        uint64_t fence_val_ = 0;
        Win32UniqueHandle fence_event_;
        Win32UniqueHandle shared_fence_handle_;

        std::list<std::tuple<ComPtr<ID3D12DeviceChild>, uint64_t>> stall_resources_;

        ComPtr<ID3D12CommandSignature> dispatch_indirect_signature_;
    };

    D3D12_DEFINE_IMP(System)
} // namespace AIHoloImager
