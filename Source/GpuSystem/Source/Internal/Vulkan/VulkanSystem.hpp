// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include <functional>

#include <volk.h>

#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandListInternal.hpp"
#include "../GpuSystemInternal.hpp"
#include "VulkanDescriptorAllocator.hpp"
#include "VulkanImpDefine.hpp"

namespace AIHoloImager
{
    class VulkanSystem : public GpuSystemInternal
    {
    public:
        explicit VulkanSystem(GpuSystem& gpu_system_, std::function<bool(GpuSystem::Api api, void* device)> confirm_device,
            bool enable_sharing, bool enable_debug);
        ~VulkanSystem() override;

        VulkanSystem(VulkanSystem&& other) noexcept;
        explicit VulkanSystem(GpuSystemInternal&& other) noexcept;
        VulkanSystem& operator=(VulkanSystem&& other) noexcept;
        GpuSystemInternal& operator=(GpuSystemInternal&& other) noexcept override;

        VkPhysicalDevice PhysicalDevice() const noexcept;
        VkDevice Device() const noexcept;
        void* NativeDevice() const noexcept override;

        LUID DeviceLuid() const noexcept override;

        uint32_t& QueueFamilyIndex(GpuSystem::CmdQueueType type) noexcept;
        uint32_t QueueFamilyIndex(GpuSystem::CmdQueueType type) const noexcept;

        uint32_t ConstantDataAlignment() const noexcept override;
        uint32_t StructuredDataAlignment() const noexcept override;
        uint32_t TextureDataAlignment() const noexcept override;
        float TimestampFrequency() const noexcept;

        void HandleDeviceLost() override;
        void ClearStallResources() override;

        void Recycle(VkBuffer buff, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkBufferView buff_view, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkImage image, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkImageView image_view, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkCommandPool cmd_pool, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkDescriptorPool desc_pool, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkDescriptorSet desc_set, VkDescriptorPool desc_pool, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkSampler sampler, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkPipelineLayout layout, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkDescriptorSetLayout layout, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkPipeline pipeline, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkDeviceMemory memory, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkRenderPass render_pass, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkQueryPool query_pool, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkSemaphore semaphore, std::shared_ptr<GpuSystem::WaitFences> wait_fences);
        void Recycle(VkQueue queue, std::shared_ptr<GpuSystem::WaitFences> wait_fences);

        uint32_t MemoryTypeIndex(uint32_t type_bits, VkMemoryPropertyFlags properties) const;
        float MaxExtraPrimitiveOverestimationSize() const noexcept;

        VulkanRecyclableObject<VkDescriptorSet>& AllocDescSet(VkDescriptorSetLayout layout);
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

        std::unique_ptr<GpuConstantBufferViewInternal> CreateConstantBufferView(const GpuMemoryBlock& mem_block) const override;

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

        std::unique_ptr<GpuTimerQueryInternal> CreateTimerQuery() const override;

        std::unique_ptr<GpuFenceInternal> CreateFence(uint64_t init_val, bool enable_sharing) const override;

        std::unique_ptr<GpuCommandQueueInternal> CreateCommandQueue(GpuSystem::CmdQueueType type, std::string_view name) const override;

    private:
        static VKAPI_ATTR VkBool32 VKAPI_CALL DebugMessageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
            VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data);

    private:
        VkInstance instance_ = VK_NULL_HANDLE;
        std::vector<std::string> supported_instance_exts_;
        VkDebugUtilsMessengerEXT debug_utils_messenger_ = VK_NULL_HANDLE;

        VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
        VkPhysicalDeviceIDProperties device_id_props_{};
        VkPhysicalDeviceProperties2 device_props_{};
        VkPhysicalDeviceMemoryProperties mem_props_{};
        float max_extra_primitive_overestimation_size_ = 0;

        VkDevice device_ = VK_NULL_HANDLE;
        uint32_t queue_family_indices_[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)];

        struct StallResourceInfo
        {
            ~StallResourceInfo()
            {
                destroy_func();
            }

            void* resource;
            std::function<void()> destroy_func;
            std::shared_ptr<GpuSystem::WaitFences> wait_fences;
        };
        std::list<StallResourceInfo> stall_resources_;

        VulkanDescriptorSetAllocator desc_set_allocators_;
    };

    VULKAN_DEFINE_IMP(System)
} // namespace AIHoloImager
