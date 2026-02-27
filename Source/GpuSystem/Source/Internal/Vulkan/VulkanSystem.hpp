// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include <functional>

#include <volk.h>

#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandListInternal.hpp"
#include "../GpuSystemInternal.hpp"
#include "VulkanCommandList.hpp"
#include "VulkanDescriptorAllocator.hpp"
#include "VulkanImpDefine.hpp"

namespace AIHoloImager
{
    class VulkanSystem : public GpuSystemInternal
    {
    public:
        VulkanSystem(GpuSystem& gpu_system_, std::function<bool(GpuSystem::Api api, void* device)> confirm_device = nullptr,
            bool enable_sharing = false, bool enable_debug = false);
        ~VulkanSystem() override;

        VulkanSystem(VulkanSystem&& other) noexcept;
        explicit VulkanSystem(GpuSystemInternal&& other) noexcept;
        VulkanSystem& operator=(VulkanSystem&& other) noexcept;
        GpuSystemInternal& operator=(GpuSystemInternal&& other) noexcept override;

        VkPhysicalDevice PhysicalDevice() const noexcept;
        VkDevice Device() const noexcept;
        void* NativeDevice() const noexcept override;
        template <typename Traits>
        typename Traits::DeviceType NativeDevice() const noexcept
        {
            return reinterpret_cast<typename Traits::DeviceType>(this->NativeDevice());
        }
        VkQueue CommandQueue(GpuSystem::CmdQueueType type) const noexcept;
        void* NativeCommandQueue(GpuSystem::CmdQueueType type) const noexcept override;
        template <typename Traits>
        typename Traits::CommandQueueType NativeCommandQueue() const noexcept
        {
            return reinterpret_cast<typename Traits::CommandQueueType>(this->NativeCommandQueue());
        }

        LUID DeviceLuid() const noexcept override;

        uint32_t& QueueFamilyIndex(GpuSystem::CmdQueueType type) noexcept;
        uint32_t QueueFamilyIndex(GpuSystem::CmdQueueType type) const noexcept;

        void* SharedFenceHandle() const noexcept override;

        [[nodiscard]] GpuCommandList CreateCommandList(GpuSystem::CmdQueueType type) override;
        uint64_t Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value) override;
        uint64_t ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value) override;
        uint64_t ExecuteAndReset(VulkanCommandList& cmd_list, uint64_t wait_fence_value);

        uint32_t ConstantDataAlignment() const noexcept override;
        uint32_t StructuredDataAlignment() const noexcept override;
        uint32_t TextureDataAlignment() const noexcept override;

        void CpuWait(uint64_t fence_value) override;
        void GpuWait(GpuSystem::CmdQueueType type, uint64_t fence_value) override;
        uint64_t FenceValue() const noexcept override;
        uint64_t CompletedFenceValue() const override;

        void HandleDeviceLost() override;
        void ClearStallResources() override;

        void Recycle(VkBuffer buff);
        void Recycle(VkBufferView buff_view);
        void Recycle(VkImage image);
        void Recycle(VkImageView image_view);
        void Recycle(VkCommandPool cmd_pool);
        void Recycle(VkDescriptorPool desc_pool);
        void Recycle(VkDescriptorSet desc_set, VkDescriptorPool desc_pool);
        void Recycle(VkSampler sampler);
        void Recycle(VkPipelineLayout layout);
        void Recycle(VkDescriptorSetLayout layout);
        void Recycle(VkPipeline pipeline);
        void Recycle(VkDeviceMemory memory);
        void Recycle(VkRenderPass render_pass);

        uint32_t MemoryTypeIndex(uint32_t type_bits, VkMemoryPropertyFlags properties) const;

        VkDescriptorSet AllocDescSet(VkDescriptorSetLayout layout);
        void DeallocDescSet(VkDescriptorSet desc_set);

        std::unique_ptr<GpuBufferInternal> CreateBuffer(
            uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name) const override;

        std::unique_ptr<GpuTextureInternal> CreateTexture(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
            uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::string_view name) const override;

        std::unique_ptr<GpuStaticSamplerInternal> CreateStaticSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const override;
        std::unique_ptr<GpuDynamicSamplerInternal> CreateDynamicSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const override;

        std::unique_ptr<GpuVertexLayoutInternal> CreateVertexLayout(
            std::span<const GpuVertexAttrib> layout, std::span<const uint32_t> slot_strides) const override;

        std::unique_ptr<GpuConstantBufferViewInternal> CreateConstantBufferView(
            const GpuBuffer& buffer, uint32_t offset, uint32_t size) const override;

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
            std::span<const ShaderInfo> shaders, const GpuVertexLayout& vertex_layout, std::span<const GpuStaticSampler> static_samplers,
            const GpuRenderPipeline::States& states) const override;
        std::unique_ptr<GpuComputePipelineInternal> CreateComputePipeline(
            const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers) const override;

        std::unique_ptr<GpuCommandPoolInternal> CreateCommandPool(GpuSystem::CmdQueueType type) const override;

        std::unique_ptr<GpuCommandListInternal> CreateCommandList(GpuCommandPool& cmd_pool, GpuSystem::CmdQueueType type) const override;

    private:
        struct CmdQueue
        {
            VkQueue cmd_queue = VK_NULL_HANDLE;
            std::vector<std::unique_ptr<GpuCommandPool>> cmd_pools;
            std::list<GpuCommandList> free_cmd_lists;
        };

    private:
        CmdQueue& GetOrCreateCommandQueue(GpuSystem::CmdQueueType type);
        CmdQueue* GetCommandQueue(GpuSystem::CmdQueueType type);
        const CmdQueue* GetCommandQueue(GpuSystem::CmdQueueType type) const;
        GpuCommandPool& CurrentCommandPool(GpuSystem::CmdQueueType type);
        uint64_t ExecuteOnly(VulkanCommandList& cmd_list, uint64_t wait_fence_value);
        static VKAPI_ATTR VkBool32 VKAPI_CALL DebugMessageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
            VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data);

    private:
        GpuSystem* gpu_system_ = nullptr;

        VkInstance instance_ = VK_NULL_HANDLE;
        std::vector<std::string> supported_instance_exts_;
        VkDebugUtilsMessengerEXT debug_utils_messenger_ = VK_NULL_HANDLE;

        VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
        VkPhysicalDeviceIDProperties device_id_props_{};
        VkPhysicalDeviceProperties2 device_props_{};
        VkPhysicalDeviceMemoryProperties mem_props_{};

        VkDevice device_ = VK_NULL_HANDLE;
        std::vector<std::string> supported_exts_{};
        uint32_t queue_family_indices_[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)];

        CmdQueue cmd_queues_[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)];

        VkSemaphore timeline_semaphore_ = VK_NULL_HANDLE;
        uint64_t fence_val_ = 0;
        Win32UniqueHandle shared_fence_handle_;

        struct StallResourceInfo
        {
            ~StallResourceInfo()
            {
                destroy_func();
            }

            void* resource;
            uint64_t fence;
            std::function<void()> destroy_func;
        };
        std::list<StallResourceInfo> stall_resources_;

        VulkanDescriptorSetAllocator desc_set_allocators_;
    };

    VULKAN_DEFINE_IMP(System)
} // namespace AIHoloImager
