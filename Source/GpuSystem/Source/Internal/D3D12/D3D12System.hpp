// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <functional>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>
#include <directx/d3d12shader.h>
#include <dxcapi.h>

#include "Base/ComPtr.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandListInternal.hpp"
#include "../GpuSystemInternal.hpp"
#include "D3D12CommandList.hpp"
#include "D3D12DescriptorAllocator.hpp"
#include "D3D12ImpDefine.hpp"

namespace AIHoloImager
{
    class D3D12System : public GpuSystemInternal
    {
    public:
        D3D12System(GpuSystem& gpu_system_, std::function<bool(GpuSystem::Api api, void* device)> confirm_device = nullptr,
            bool enable_sharing = false, bool enable_debug = false);
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

        LUID DeviceLuid() const noexcept override;

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
        ComPtr<ID3D12ShaderReflection> ShaderReflect(std::span<const uint8_t> bytecode);

        uint32_t RtvDescSize() const noexcept;
        uint32_t DsvDescSize() const noexcept;
        uint32_t CbvSrvUavDescSize() const noexcept;
        uint32_t SamplerDescSize() const noexcept;

        std::unique_ptr<D3D12DescriptorHeap> CreateDescriptorHeap(
            uint32_t size, D3D12_DESCRIPTOR_HEAP_TYPE type, bool shader_visible, std::string_view name) const;

        uint32_t DescriptorSize(D3D12_DESCRIPTOR_HEAP_TYPE type) const;

        D3D12DescriptorBlock AllocRtvDescBlock(uint32_t size);
        void DeallocRtvDescBlock(D3D12DescriptorBlock&& desc_block);
        D3D12DescriptorBlock AllocDsvDescBlock(uint32_t size);
        void DeallocDsvDescBlock(D3D12DescriptorBlock&& desc_block);
        D3D12DescriptorBlock AllocCbvSrvUavDescBlock(uint32_t size);
        void DeallocCbvSrvUavDescBlock(D3D12DescriptorBlock&& desc_block);
        D3D12DescriptorBlock AllocShaderVisibleCbvSrvUavDescBlock(uint32_t size);
        void DeallocShaderVisibleCbvSrvUavDescBlock(D3D12DescriptorBlock&& desc_block);
        D3D12DescriptorBlock AllocSamplerDescBlock(uint32_t size);
        void DeallocSamplerDescBlock(D3D12DescriptorBlock&& desc_block);
        D3D12DescriptorBlock AllocShaderVisibleSamplerDescBlock(uint32_t size);
        void DeallocShaderVisibleSamplerDescBlock(D3D12DescriptorBlock&& desc_block);

        std::unique_ptr<GpuBufferInternal> CreateBuffer(
            uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name) const override;
        std::unique_ptr<GpuBufferInternal> CreateBuffer(
            void* native_resource, GpuResourceState curr_state, std::string_view name) const override;

        std::unique_ptr<GpuTextureInternal> CreateTexture(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
            uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::string_view name) const override;
        std::unique_ptr<GpuTextureInternal> CreateTexture(
            void* native_resource, GpuResourceState curr_state, std::string_view name) const override;

        std::unique_ptr<GpuStaticSamplerInternal> CreateStaticSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const override;
        std::unique_ptr<GpuDynamicSamplerInternal> CreateDynamicSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const override;

        std::unique_ptr<GpuVertexAttribsInternal> CreateVertexAttribs(
            std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides) const override;

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
        static void DebugMessageCallback(
            D3D12_MESSAGE_CATEGORY category, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id, LPCSTR description, void* context);

    private:
        GpuSystem* gpu_system_ = nullptr;

        ComPtr<ID3D12Device> device_;
        DWORD dbg_callback_cookie_ = 0;

        CmdQueue cmd_queues_[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)];

        ComPtr<ID3D12Fence> fence_;
        uint64_t fence_val_ = 0;
        Win32UniqueHandle fence_event_;
        Win32UniqueHandle shared_fence_handle_;

        std::list<std::tuple<ComPtr<ID3D12DeviceChild>, uint64_t>> stall_resources_;

        ComPtr<ID3D12CommandSignature> dispatch_indirect_signature_;

        ComPtr<IDxcUtils> dxc_utils_;

        D3D12DescriptorAllocator rtv_desc_allocator_;
        D3D12DescriptorAllocator dsv_desc_allocator_;
        D3D12DescriptorAllocator cbv_srv_uav_desc_allocator_;
        D3D12DescriptorAllocator shader_visible_cbv_srv_uav_desc_allocator_;
        D3D12DescriptorAllocator sampler_desc_allocator_;
        D3D12DescriptorAllocator shader_visible_sampler_desc_allocator_;
    };

    D3D12_DEFINE_IMP(System)
} // namespace AIHoloImager
