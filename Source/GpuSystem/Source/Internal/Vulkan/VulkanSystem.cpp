// Copyright (c) 2025-2026 Minmin Gong
//

#include "VulkanSystem.hpp"

#include <algorithm>
#include <cassert>
#include <format>
#include <iostream>
#include <limits>
#include <set>

#include "Base/ErrorHandling.hpp"

#include "VulkanBuffer.hpp"
#include "VulkanCommandList.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanCommandQueue.hpp"
#include "VulkanErrorHandling.hpp"
#include "VulkanFence.hpp"
#include "VulkanQuery.hpp"
#include "VulkanResourceViews.hpp"
#include "VulkanSampler.hpp"
#include "VulkanShader.hpp"
#include "VulkanTexture.hpp"
#include "VulkanVertexLayout.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(System)

    VulkanSystem::VulkanSystem(
        GpuSystem& gpu_system, std::function<bool(GpuSystem::Api api, void* device)> confirm_device, bool enable_sharing, bool enable_debug)
        : GpuSystemInternal(gpu_system, enable_sharing)
    {
        std::fill(std::begin(queue_family_indices_), std::end(queue_family_indices_), std::numeric_limits<uint32_t>::max());

        TIFVK(volkInitialize());

        std::vector<const char*> instance_exts = {VK_KHR_SURFACE_EXTENSION_NAME};
        instance_exts.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);

        uint32_t instance_ext_count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &instance_ext_count, nullptr);
        if (instance_ext_count > 0)
        {
            std::vector<VkExtensionProperties> extensions(instance_ext_count);
            if (vkEnumerateInstanceExtensionProperties(nullptr, &instance_ext_count, extensions.data()) == VK_SUCCESS)
            {
                for (auto& extension : extensions)
                {
                    supported_instance_exts_.push_back(extension.extensionName);
                }
            }
        }

        const VkApplicationInfo app_info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "GpuSystem",
            .pEngineName = "GpuSystem",
            .apiVersion = VK_API_VERSION_1_3,
        };

        VkInstanceCreateInfo instance_create_info{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
        };

        VkDebugUtilsMessengerCreateInfoEXT debug_utils_msg_create_info;
        if (enable_debug)
        {
            debug_utils_msg_create_info = {
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                .pNext = instance_create_info.pNext,
                .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                .pfnUserCallback = VulkanSystem::DebugMessageCallback,
            };

            instance_create_info.pNext = &debug_utils_msg_create_info;
        }

        if (enable_debug || (std::find(supported_instance_exts_.begin(), supported_instance_exts_.end(),
                                 VK_EXT_DEBUG_UTILS_EXTENSION_NAME) != supported_instance_exts_.end()))
        {
            instance_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        if (!instance_exts.empty())
        {
            instance_create_info.enabledExtensionCount = static_cast<uint32_t>(instance_exts.size());
            instance_create_info.ppEnabledExtensionNames = instance_exts.data();
        }

        const char* validation_layer_name = "VK_LAYER_KHRONOS_validation";
        if (enable_debug)
        {
            uint32_t instance_layer_count;
            vkEnumerateInstanceLayerProperties(&instance_layer_count, nullptr);
            std::vector<VkLayerProperties> instance_layer_props(instance_layer_count);
            vkEnumerateInstanceLayerProperties(&instance_layer_count, instance_layer_props.data());
            bool validation_layer_present = false;
            for (auto& layer : instance_layer_props)
            {
                if (std::string_view(layer.layerName) == validation_layer_name)
                {
                    validation_layer_present = true;
                    break;
                }
            }
            if (validation_layer_present)
            {
                instance_create_info.ppEnabledLayerNames = &validation_layer_name;
                instance_create_info.enabledLayerCount = 1;
            }
            else
            {
                std::cerr << "Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled.\n";
            }
        }

        TIFVK(vkCreateInstance(&instance_create_info, nullptr, &instance_));
        volkLoadInstance(instance_);

        if (enable_debug)
        {
            debug_utils_msg_create_info.pNext = nullptr;
            TIFVK(vkCreateDebugUtilsMessengerEXT(instance_, &debug_utils_msg_create_info, nullptr, &debug_utils_messenger_));
        }

        uint32_t gpu_count = 0;
        TIFVK(vkEnumeratePhysicalDevices(instance_, &gpu_count, nullptr));
        if (gpu_count == 0)
        {
            std::cerr << "No device with Vulkan support found\n";
            return;
        }

        std::vector<VkPhysicalDevice> physical_devices(gpu_count);
        TIFVK(vkEnumeratePhysicalDevices(instance_, &gpu_count, physical_devices.data()));

        const auto support_necessary_features = [](VkPhysicalDevice physical_device) -> bool {
            VkPhysicalDeviceRobustness2FeaturesKHR robustness_2_feature{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_KHR,
            };

            VkPhysicalDeviceVulkan11Features vulkan11_features = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                .pNext = &robustness_2_feature,
            };

            VkPhysicalDeviceVulkan12Features vulkan12_features = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                .pNext = &vulkan11_features,
            };

            VkPhysicalDeviceVulkan13Features vulkan13_features = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                .pNext = &vulkan12_features,
            };

            VkPhysicalDeviceFeatures2 physical_features{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = &vulkan13_features,
            };

            vkGetPhysicalDeviceFeatures2(physical_device, &physical_features);

            const VkBool32 requires_bits[] = {
                physical_features.features.independentBlend,
                physical_features.features.geometryShader,
                robustness_2_feature.nullDescriptor,
                vulkan11_features.storageBuffer16BitAccess,
                vulkan11_features.uniformAndStorageBuffer16BitAccess,
                vulkan12_features.shaderFloat16,
                vulkan12_features.timelineSemaphore,
                vulkan13_features.shaderDemoteToHelperInvocation,
                vulkan13_features.dynamicRendering,
            };

            return std::all_of(std::begin(requires_bits), std::end(requires_bits), [](VkBool32 bit) { return bit == VK_TRUE; });
        };

        std::vector<const char*> enable_device_exts = {
            VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
            VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME,
        };
        if (enable_sharing)
        {
            enable_device_exts.insert(enable_device_exts.end(), {
                                                                    VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
                                                                    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
                                                                });
        }

        const auto support_required_extensions = [&enable_device_exts](VkPhysicalDevice physical_device) -> bool {
            uint32_t device_ext_count = 0;
            vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_ext_count, nullptr);
            if (device_ext_count == 0)
            {
                return false;
            }

            std::vector<VkExtensionProperties> supported_exts(device_ext_count);
            if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_ext_count, supported_exts.data()) != VK_SUCCESS)
            {
                return false;
            }

            for (const auto* required_ext : enable_device_exts)
            {
                auto iter = std::find_if(supported_exts.begin(), supported_exts.end(),
                    [required_ext](const VkExtensionProperties& ext) { return std::string_view(required_ext) == ext.extensionName; });
                if (iter == supported_exts.end())
                {
                    return false;
                }
            }

            return true;
        };

        physical_device_ = VK_NULL_HANDLE;
        for (auto& physical_device : physical_devices)
        {
            if (support_necessary_features(physical_device) && support_required_extensions(physical_device) &&
                (!confirm_device || confirm_device(GpuSystem::Api::Vulkan, reinterpret_cast<void*>(physical_device))))
            {
                physical_device_ = physical_device;
                break;
            }
        }
        Verify(physical_device_ != VK_NULL_HANDLE);

        VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservative_raster_props{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT,
        };
        device_id_props_ = VkPhysicalDeviceIDProperties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
            .pNext = &conservative_raster_props,
        };
        device_props_ = VkPhysicalDeviceProperties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &device_id_props_,
        };
        vkGetPhysicalDeviceProperties2(physical_device_, &device_props_);
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props_);
        max_extra_primitive_overestimation_size_ = conservative_raster_props.maxExtraPrimitiveOverestimationSize;

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
        assert(queue_family_count > 0);
        auto queue_family_props = std::make_unique<VkQueueFamilyProperties[]>(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_family_props.get());

        std::set<uint32_t> queue_family_indices;
        for (uint32_t i = 0; i < queue_family_count; ++i)
        {
            if ((this->QueueFamilyIndex(GpuSystem::CmdQueueType::Render) == std::numeric_limits<uint32_t>::max()) &&
                (queue_family_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            {
                this->QueueFamilyIndex(GpuSystem::CmdQueueType::Render) = i;
                queue_family_indices.insert(i);
            }
            if ((this->QueueFamilyIndex(GpuSystem::CmdQueueType::Compute) == std::numeric_limits<uint32_t>::max()) &&
                (queue_family_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(queue_family_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            {
                this->QueueFamilyIndex(GpuSystem::CmdQueueType::Compute) = i;
                queue_family_indices.insert(i);
            }
            if ((this->QueueFamilyIndex(GpuSystem::CmdQueueType::Copy) == std::numeric_limits<uint32_t>::max()) &&
                (queue_family_props[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(queue_family_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                !(queue_family_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            {
                this->QueueFamilyIndex(GpuSystem::CmdQueueType::Copy) = i;
                queue_family_indices.insert(i);
            }
            if ((this->QueueFamilyIndex(GpuSystem::CmdQueueType::VideoEncode) == std::numeric_limits<uint32_t>::max()) &&
                (queue_family_props[i].queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR))
            {
                this->QueueFamilyIndex(GpuSystem::CmdQueueType::VideoEncode) = i;
                queue_family_indices.insert(i);
            }
        }

        std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
        const float default_queue_priority = 0;
        for (const auto& index : queue_family_indices)
        {
            queue_create_infos.emplace_back(VkDeviceQueueCreateInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = index,
                .queueCount = 1,
                .pQueuePriorities = &default_queue_priority,
            });
        }

        const VkPhysicalDeviceFeatures enable_features{
            .independentBlend = VK_TRUE,
            .geometryShader = VK_TRUE,
        };

        VkPhysicalDeviceRobustness2FeaturesKHR enable_robustness_2_feature{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_KHR,
            .nullDescriptor = VK_TRUE,
        };

        VkPhysicalDeviceVulkan11Features enable_vulkan11_features = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .pNext = &enable_robustness_2_feature,
            .storageBuffer16BitAccess = VK_TRUE,
            .uniformAndStorageBuffer16BitAccess = VK_TRUE,
        };

        VkPhysicalDeviceVulkan12Features enable_vulkan12_features = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = &enable_vulkan11_features,
            .shaderFloat16 = VK_TRUE,
            .timelineSemaphore = VK_TRUE,
        };

        VkPhysicalDeviceVulkan13Features enable_vulkan13_features = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &enable_vulkan12_features,
            .shaderDemoteToHelperInvocation = VK_TRUE,
            .dynamicRendering = VK_TRUE,
        };

        const VkDeviceCreateInfo device_create_info{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &enable_vulkan13_features,
            .queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size()),
            .pQueueCreateInfos = queue_create_infos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(enable_device_exts.size()),
            .ppEnabledExtensionNames = enable_device_exts.data(),
            .pEnabledFeatures = &enable_features,
        };

        TIFVK(vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_));
        volkLoadDevice(device_);
    }

    VulkanSystem::~VulkanSystem()
    {
        this->CpuWait(GpuSystem::WaitFences::Forever());

        desc_set_allocators_ = VulkanDescriptorSetAllocator();

        this->ClearCommandQueueContexts();

        stall_resources_.clear();

        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;

        if (debug_utils_messenger_ != VK_NULL_HANDLE)
        {
            vkDestroyDebugUtilsMessengerEXT(instance_, debug_utils_messenger_, nullptr);
        }

        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    VulkanSystem::VulkanSystem(VulkanSystem&& other) noexcept = default;
    VulkanSystem::VulkanSystem(GpuSystemInternal&& other) noexcept : VulkanSystem(static_cast<VulkanSystem&&>(other))
    {
    }
    VulkanSystem& VulkanSystem::operator=(VulkanSystem&& other) noexcept = default;
    GpuSystemInternal& VulkanSystem::operator=(GpuSystemInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanSystem&&>(other));
    }

    VkPhysicalDevice VulkanSystem::PhysicalDevice() const noexcept
    {
        return physical_device_;
    }

    VkDevice VulkanSystem::Device() const noexcept
    {
        return device_;
    }

    void* VulkanSystem::NativeDevice() const noexcept
    {
        return this->Device();
    }

    uint32_t& VulkanSystem::QueueFamilyIndex(GpuSystem::CmdQueueType type) noexcept
    {
        return queue_family_indices_[static_cast<uint32_t>(type)];
    }

    uint32_t VulkanSystem::QueueFamilyIndex(GpuSystem::CmdQueueType type) const noexcept
    {
        return const_cast<VulkanSystem*>(this)->QueueFamilyIndex(type);
    }

    LUID VulkanSystem::DeviceLuid() const noexcept
    {
        LUID ret{};
        if (device_id_props_.deviceLUIDValid)
        {
            std::memcpy(&ret, device_id_props_.deviceLUID, sizeof(ret));
        }
        return ret;
    }

    uint32_t VulkanSystem::ConstantDataAlignment() const noexcept
    {
        return static_cast<uint32_t>(device_props_.properties.limits.minUniformBufferOffsetAlignment);
    }
    uint32_t VulkanSystem::StructuredDataAlignment() const noexcept
    {
        return static_cast<uint32_t>(device_props_.properties.limits.minStorageBufferOffsetAlignment);
    }
    uint32_t VulkanSystem::TextureDataAlignment() const noexcept
    {
        return static_cast<uint32_t>(device_props_.properties.limits.optimalBufferCopyOffsetAlignment);
    }
    float VulkanSystem::TimestampFrequency() const noexcept
    {
        return device_props_.properties.limits.timestampPeriod;
    }

    void VulkanSystem::HandleDeviceLost()
    {
        this->ClearCommandQueueContexts();

        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    void VulkanSystem::Recycle(VkBuffer buff, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(buff, [this, buff]() { vkDestroyBuffer(device_, buff, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkBufferView buff_view, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            buff_view, [this, buff_view]() { vkDestroyBufferView(device_, buff_view, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkImage image, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(image, [this, image]() { vkDestroyImage(device_, image, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkImageView image_view, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            image_view, [this, image_view]() { vkDestroyImageView(device_, image_view, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkCommandPool cmd_pool, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            cmd_pool, [this, cmd_pool]() { vkDestroyCommandPool(device_, cmd_pool, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkDescriptorPool desc_pool, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            desc_pool, [this, desc_pool]() { vkDestroyDescriptorPool(device_, desc_pool, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkDescriptorSet desc_set, VkDescriptorPool desc_pool, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            desc_set, [this, desc_set, desc_pool]() { vkFreeDescriptorSets(device_, desc_pool, 1, &desc_set); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkSampler sampler, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(sampler, [this, sampler]() { vkDestroySampler(device_, sampler, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkPipelineLayout layout, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            layout, [this, layout]() { vkDestroyPipelineLayout(device_, layout, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkDescriptorSetLayout layout, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            layout, [this, layout]() { vkDestroyDescriptorSetLayout(device_, layout, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkPipeline pipeline, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            pipeline, [this, pipeline]() { vkDestroyPipeline(device_, pipeline, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkDeviceMemory memory, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(memory, [this, memory]() { vkFreeMemory(device_, memory, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkRenderPass render_pass, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            render_pass, [this, render_pass]() { vkDestroyRenderPass(device_, render_pass, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkQueryPool query_pool, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            query_pool, [this, query_pool]() { vkDestroyQueryPool(device_, query_pool, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkSemaphore semaphore, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(
            semaphore, [this, semaphore]() { vkDestroySemaphore(device_, semaphore, nullptr); }, std::move(wait_fences));
    }
    void VulkanSystem::Recycle(VkQueue queue, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(queue, []() {}, std::move(wait_fences));
    }

    void VulkanSystem::ClearStallResources()
    {
        for (uint32_t i = 0; i < static_cast<uint32_t>(GpuSystem::CmdQueueType::Num); ++i)
        {
            const auto queue_type = static_cast<GpuSystem::CmdQueueType>(i);
            const uint64_t completed_fence = this->CompletedFenceValue(queue_type);
            if (completed_fence != 0)
            {
                for (auto res_iter = stall_resources_.begin(); res_iter != stall_resources_.end(); ++res_iter)
                {
                    if (res_iter->wait_fences->fence_values[i] <= completed_fence)
                    {
                        res_iter->wait_fences->fence_values[i] = 0;
                    }
                }
            }
        }

        for (auto iter = stall_resources_.begin(); iter != stall_resources_.end();)
        {
            bool all_completed = true;
            for (uint32_t i = 0; i < static_cast<uint32_t>(GpuSystem::CmdQueueType::Num); ++i)
            {
                if (iter->wait_fences->fence_values[i] != 0)
                {
                    all_completed = false;
                    break;
                }
            }

            if (all_completed)
            {
                iter = stall_resources_.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }

    std::unique_ptr<GpuBufferInternal> VulkanSystem::CreateBuffer(
        uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name) const
    {
        return std::make_unique<VulkanBuffer>(this->GpuSys(), size, heap, flags, std::move(name));
    }

    std::unique_ptr<GpuTextureInternal> VulkanSystem::CreateTexture(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
        uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::string_view name) const
    {
        return std::make_unique<VulkanTexture>(
            this->GpuSys(), type, width, height, depth, array_size, mip_levels, format, flags, std::move(name));
    }

    std::unique_ptr<GpuStaticSamplerInternal> VulkanSystem::CreateStaticSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<VulkanStaticSampler>(this->GpuSys(), filters, addr_modes);
    }

    std::unique_ptr<GpuDynamicSamplerInternal> VulkanSystem::CreateDynamicSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<VulkanDynamicSampler>(this->GpuSys(), filters, addr_modes);
    }

    std::unique_ptr<GpuVertexLayoutInternal> VulkanSystem::CreateVertexLayout(
        std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides) const
    {
        return std::make_unique<VulkanVertexLayout>(std::move(attribs), std::move(slot_strides));
    }

    std::unique_ptr<GpuConstantBufferViewInternal> VulkanSystem::CreateConstantBufferView(const GpuMemoryBlock& mem_block) const
    {
        return std::make_unique<VulkanConstantBufferView>(mem_block);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> VulkanSystem::CreateShaderResourceView(
        const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanShaderResourceView>(this->GpuSys(), texture, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> VulkanSystem::CreateShaderResourceView(
        const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanShaderResourceView>(this->GpuSys(), texture_array, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> VulkanSystem::CreateShaderResourceView(
        const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanShaderResourceView>(this->GpuSys(), texture, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> VulkanSystem::CreateShaderResourceView(
        const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const
    {
        return std::make_unique<VulkanShaderResourceView>(this->GpuSys(), buffer, first_element, num_elements, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> VulkanSystem::CreateShaderResourceView(
        const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const
    {
        return std::make_unique<VulkanShaderResourceView>(this->GpuSys(), buffer, first_element, num_elements, element_size);
    }

    std::unique_ptr<GpuRenderTargetViewInternal> VulkanSystem::CreateRenderTargetView(GpuTexture2D& texture, GpuFormat format) const
    {
        return std::make_unique<VulkanRenderTargetView>(this->GpuSys(), texture, format);
    }

    std::unique_ptr<GpuDepthStencilViewInternal> VulkanSystem::CreateDepthStencilView(GpuTexture2D& texture, GpuFormat format) const
    {
        return std::make_unique<VulkanDepthStencilView>(this->GpuSys(), texture, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> VulkanSystem::CreateUnorderedAccessView(
        GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanUnorderedAccessView>(this->GpuSys(), texture, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> VulkanSystem::CreateUnorderedAccessView(
        GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanUnorderedAccessView>(this->GpuSys(), texture_array, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> VulkanSystem::CreateUnorderedAccessView(
        GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanUnorderedAccessView>(this->GpuSys(), texture, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> VulkanSystem::CreateUnorderedAccessView(
        GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const
    {
        return std::make_unique<VulkanUnorderedAccessView>(this->GpuSys(), buffer, first_element, num_elements, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> VulkanSystem::CreateUnorderedAccessView(
        GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const
    {
        return std::make_unique<VulkanUnorderedAccessView>(this->GpuSys(), buffer, first_element, num_elements, element_size);
    }

    std::unique_ptr<GpuRenderPipelineInternal> VulkanSystem::CreateRenderPipeline(GpuRenderPipeline::PrimitiveTopology topology,
        std::span<const ShaderInfo> shaders, const GpuVertexLayout& vertex_layout, std::span<const GpuStaticSampler> static_samplers,
        const GpuRenderPipeline::States& states) const
    {
        return std::make_unique<VulkanRenderPipeline>(
            this->GpuSys(), topology, std::move(shaders), vertex_layout, std::move(static_samplers), states);
    }

    std::unique_ptr<GpuComputePipelineInternal> VulkanSystem::CreateComputePipeline(
        const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers) const
    {
        return std::make_unique<VulkanComputePipeline>(this->GpuSys(), shader, std::move(static_samplers));
    }

    std::unique_ptr<GpuCommandPoolInternal> VulkanSystem::CreateCommandPool(GpuSystem::CmdQueueType type) const
    {
        return std::make_unique<VulkanCommandPool>(this->GpuSys(), type);
    }

    std::unique_ptr<GpuCommandListInternal> VulkanSystem::CreateCommandList(GpuCommandPool& cmd_pool, GpuSystem::CmdQueueType type) const
    {
        return std::make_unique<VulkanCommandList>(this->GpuSys(), cmd_pool, type);
    }

    std::unique_ptr<GpuTimerQueryInternal> VulkanSystem::CreateTimerQuery() const
    {
        return std::make_unique<VulkanTimerQuery>(this->GpuSys());
    }

    std::unique_ptr<GpuFenceInternal> VulkanSystem::CreateFence(uint64_t init_val, bool enable_sharing) const
    {
        return std::make_unique<VulkanFence>(this->GpuSys(), init_val, enable_sharing);
    }

    std::unique_ptr<GpuCommandQueueInternal> VulkanSystem::CreateCommandQueue(GpuSystem::CmdQueueType type, std::string_view name) const
    {
        return std::make_unique<VulkanCommandQueue>(this->GpuSys(), type, std::move(name));
    }

    VkBool32 VulkanSystem::DebugMessageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
        [[maybe_unused]] void* user_data)
    {
        constexpr const char* RedEscape = "\033[31m";
        constexpr const char* GreenEscape = "\033[32m";
        constexpr const char* YellowEscape = "\033[33m";
        constexpr const char* CyanEscape = "\033[36m";
        constexpr const char* EndEscape = "\033[0m";

        const char* color_escape;
        const char* severity_str;
        if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
        {
            color_escape = GreenEscape;
            severity_str = "VERBOSE";
        }
        else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        {
            color_escape = CyanEscape;
            severity_str = "INFO";
        }
        else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            color_escape = YellowEscape;
            severity_str = "WARNING";
        }
        else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        {
            color_escape = RedEscape;
            severity_str = "ERROR";
        }

        std::ostream& output_stream = (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? std::cerr : std::cout;

        output_stream << std::format("{}{}: {}[{}]", color_escape, severity_str, EndEscape, callback_data->messageIdNumber);
        if (callback_data->pMessageIdName)
        {
            output_stream << std::format("[{}]", callback_data->pMessageIdName);
        }
        output_stream << std::format(": {}\n", callback_data->pMessage);
        output_stream.flush();

        if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        {
            return VK_TRUE;
        }
        else
        {
            return VK_FALSE;
        }
    }

    uint32_t VulkanSystem::MemoryTypeIndex(uint32_t type_bits, VkMemoryPropertyFlags properties) const
    {
        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i)
        {
            if ((type_bits & 1) == 1)
            {
                if ((mem_props_.memoryTypes[i].propertyFlags & properties) == properties)
                {
                    return i;
                }
            }
            type_bits >>= 1;
        }

        return static_cast<uint32_t>(-1);
    }

    float VulkanSystem::MaxExtraPrimitiveOverestimationSize() const noexcept
    {
        return max_extra_primitive_overestimation_size_;
    }

    VulkanRecyclableObject<VkDescriptorSet>& VulkanSystem::AllocDescSet(VkDescriptorSetLayout layout)
    {
        if (!desc_set_allocators_)
        {
            desc_set_allocators_ = VulkanDescriptorSetAllocator(this->GpuSys());
        }

        return desc_set_allocators_.Allocate(layout);
    }

    void VulkanSystem::DeallocDescSet(VkDescriptorSet desc_set)
    {
        if (desc_set != VK_NULL_HANDLE)
        {
            desc_set_allocators_.Deallocate(desc_set);
        }
    }
} // namespace AIHoloImager
