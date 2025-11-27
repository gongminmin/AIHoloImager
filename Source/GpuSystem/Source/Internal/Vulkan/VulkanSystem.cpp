// Copyright (c) 2025 Minmin Gong
//

#include "VulkanSystem.hpp"

#include <cassert>
#include <format>
#include <iostream>
#include <limits>
#include <set>

#include "Base/ErrorHandling.hpp"

#include "VulkanBuffer.hpp"
#include "VulkanCommandList.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanErrorhandling.hpp"
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
        : gpu_system_(&gpu_system)
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

        const auto support_timeline_semaphore = [](VkPhysicalDevice physical_device) -> bool {
            VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            };

            VkPhysicalDeviceFeatures2 physical_features{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = &timeline_features,
            };

            vkGetPhysicalDeviceFeatures2(physical_device, &physical_features);

            return timeline_features.timelineSemaphore == VK_TRUE;
        };

        const auto support_robustness_2 = [](VkPhysicalDevice physical_device) -> bool {
            VkPhysicalDeviceRobustness2FeaturesEXT robustness_2_features{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
            };

            VkPhysicalDeviceFeatures2 physical_features{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = &robustness_2_features,
            };

            vkGetPhysicalDeviceFeatures2(physical_device, &physical_features);

            return robustness_2_features.nullDescriptor == VK_TRUE;
        };

        physical_device_ = VK_NULL_HANDLE;
        for (auto& physical_device : physical_devices)
        {
            if (support_timeline_semaphore(physical_device) && support_robustness_2(physical_device) &&
                (!confirm_device || confirm_device(GpuSystem::Api::Vulkan, reinterpret_cast<void*>(physical_device))))
            {
                physical_device_ = physical_device;
                break;
            }
        }
        Verify(physical_device_ != VK_NULL_HANDLE);

        device_id_props_ = VkPhysicalDeviceIDProperties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
        };
        device_props_ = VkPhysicalDeviceProperties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &device_id_props_,
        };
        vkGetPhysicalDeviceProperties2(physical_device_, &device_props_);
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props_);

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
        assert(queue_family_count > 0);
        auto queue_family_props = std::make_unique<VkQueueFamilyProperties[]>(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_family_props.get());

        uint32_t device_ext_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &device_ext_count, nullptr);
        if (device_ext_count > 0)
        {
            std::vector<VkExtensionProperties> extensions(device_ext_count);
            if (vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &device_ext_count, extensions.data()) == VK_SUCCESS)
            {
                for (const auto& ext : extensions)
                {
                    supported_exts_.push_back(ext.extensionName);
                }
            }
        }

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
                (queue_family_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                this->QueueFamilyIndex(GpuSystem::CmdQueueType::Compute) = i;
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
            auto& queue_info = queue_create_infos.emplace_back();
            queue_info = VkDeviceQueueCreateInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = index,
                .queueCount = 1,
                .pQueuePriorities = &default_queue_priority,
            };
        }

        const VkPhysicalDeviceFeatures enable_features{
            .geometryShader = VK_TRUE,
        };

        VkPhysicalDeviceShaderFloat16Int8Features shader_float16_feature{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
            .shaderFloat16 = VK_TRUE,
        };

        VkPhysicalDevice16BitStorageFeatures storage_16_feature{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
            .pNext = &shader_float16_feature,
            .storageBuffer16BitAccess = VK_TRUE,
            .uniformAndStorageBuffer16BitAccess = VK_TRUE,
        };

        VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_feature{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            .pNext = &storage_16_feature,
            .timelineSemaphore = VK_TRUE,
        };

        VkPhysicalDeviceRobustness2FeaturesKHR robustness_2_feature{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_KHR,
            .pNext = &timeline_semaphore_feature,
            .nullDescriptor = VK_TRUE,
        };

        VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_feature{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
            .pNext = &robustness_2_feature,
            .dynamicRendering = VK_TRUE,
        };

        VkDeviceCreateInfo device_create_info{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &dynamic_rendering_feature,
            .queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size()),
            .pQueueCreateInfos = queue_create_infos.data(),
            .pEnabledFeatures = &enable_features,
        };

        std::vector<const char*> device_exts = {
            VK_KHR_ROBUSTNESS_2_EXTENSION_NAME,
            VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME,
        };
        if (enable_sharing)
        {
            device_exts.insert(device_exts.end(), {
                                                      VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
                                                      VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
                                                  });
        }

        for (auto iter = device_exts.begin(); iter != device_exts.end();)
        {
            if (std::find(supported_exts_.begin(), supported_exts_.end(), *iter) == supported_exts_.end())
            {
                iter = device_exts.erase(iter);
            }
            else
            {
                ++iter;
            }
        }

        if (!device_exts.empty())
        {
            device_create_info.enabledExtensionCount = static_cast<uint32_t>(device_exts.size());
            device_create_info.ppEnabledExtensionNames = device_exts.data();
        }

        TIFVK(vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_));
        volkLoadDevice(device_);

        {
            VkSemaphoreTypeCreateInfo timeline_create_info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                .initialValue = fence_val_,
            };

            VkExportSemaphoreCreateInfo export_info;
            if (enable_sharing)
            {
                export_info = {
                    .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
                    .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
                };

                timeline_create_info.pNext = &export_info;
            }

            const VkSemaphoreCreateInfo semaphore_create_info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = &timeline_create_info,
                .flags = 0,
            };
            TIFVK(vkCreateSemaphore(device_, &semaphore_create_info, nullptr, &timeline_semaphore_));

            ++fence_val_;

            if (enable_sharing)
            {
                const VkSemaphoreGetWin32HandleInfoKHR get_handle_info{
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
                    .semaphore = timeline_semaphore_,
                    .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
                };

                HANDLE shared_handle;
                TIFVK(vkGetSemaphoreWin32HandleKHR(device_, &get_handle_info, &shared_handle));
                shared_fence_handle_.reset(shared_handle);
            }
        }
    }

    VulkanSystem::~VulkanSystem()
    {
        this->CpuWait(GpuSystem::MaxFenceValue);

        desc_set_allocators_ = VulkanDescriptorSetAllocator();

        for (auto& cmd_queue : cmd_queues_)
        {
            if (cmd_queue.cmd_queue != VK_NULL_HANDLE)
            {
                cmd_queue.free_cmd_lists.clear();
                cmd_queue.cmd_pools.clear();
                cmd_queue.cmd_queue = VK_NULL_HANDLE;
            }
        }

        stall_resources_.clear();

        vkDestroySemaphore(device_, timeline_semaphore_, nullptr);
        timeline_semaphore_ = VK_NULL_HANDLE;

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

    VkQueue VulkanSystem::CommandQueue(GpuSystem::CmdQueueType type) const noexcept
    {
        return cmd_queues_[static_cast<uint32_t>(type)].cmd_queue;
    }

    void* VulkanSystem::NativeCommandQueue(GpuSystem::CmdQueueType type) const noexcept
    {
        return this->CommandQueue(type);
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

    void* VulkanSystem::SharedFenceHandle() const noexcept
    {
        return shared_fence_handle_.get();
    }

    GpuCommandList VulkanSystem::CreateCommandList(GpuSystem::CmdQueueType type)
    {
        GpuCommandList cmd_list;
        auto& cmd_pool = this->CurrentCommandPool(type);
        auto& cmd_queue = this->GetOrCreateCommandQueue(type);
        if (cmd_queue.free_cmd_lists.empty())
        {
            cmd_list = GpuCommandList(*gpu_system_, cmd_pool, type);
        }
        else
        {
            cmd_list = std::move(cmd_queue.free_cmd_lists.front());
            cmd_queue.free_cmd_lists.pop_front();
            cmd_list.Reset(cmd_pool);
        }
        return cmd_list;
    }

    uint64_t VulkanSystem::Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value)
    {
        const uint64_t new_fence_value = this->ExecuteOnly(VulkanImp(cmd_list), wait_fence_value);
        this->GetOrCreateCommandQueue(cmd_list.Type()).free_cmd_lists.emplace_back(std::move(cmd_list));
        return new_fence_value;
    }

    uint64_t VulkanSystem::ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value)
    {
        return this->ExecuteAndReset(VulkanImp(cmd_list), wait_fence_value);
    }

    uint64_t VulkanSystem::ExecuteAndReset(VulkanCommandList& cmd_list, uint64_t wait_fence_value)
    {
        const uint64_t new_fence_value = this->ExecuteOnly(cmd_list, wait_fence_value);
        cmd_list.Reset(this->CurrentCommandPool(cmd_list.Type()));
        return new_fence_value;
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

    void VulkanSystem::CpuWait(uint64_t fence_value)
    {
        if (timeline_semaphore_ != VK_NULL_HANDLE)
        {
            for (auto& cmd_queue : cmd_queues_)
            {
                if (cmd_queue.cmd_queue != VK_NULL_HANDLE)
                {
                    uint64_t wait_fence_value;
                    if (fence_value == GpuSystem::MaxFenceValue)
                    {
                        wait_fence_value = fence_val_;

                        const VkTimelineSemaphoreSubmitInfo timeline_info{
                            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                            .signalSemaphoreValueCount = 1,
                            .pSignalSemaphoreValues = &fence_val_,
                        };

                        const VkSubmitInfo submit_info{
                            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .pNext = &timeline_info,
                            .signalSemaphoreCount = 1,
                            .pSignalSemaphores = &timeline_semaphore_,
                        };

                        if (vkQueueSubmit(cmd_queue.cmd_queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS)
                        {
                            continue;
                        }
                        ++fence_val_;
                    }
                    else
                    {
                        wait_fence_value = fence_value;
                    }

                    uint64_t completed_value;
                    if (vkGetSemaphoreCounterValue(device_, timeline_semaphore_, &completed_value) != VK_SUCCESS)
                    {
                        continue;
                    }
                    if (completed_value < wait_fence_value)
                    {
                        const VkSemaphoreWaitInfo wait_info{
                            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                            .flags = 0,
                            .semaphoreCount = 1,
                            .pSemaphores = &timeline_semaphore_,
                            .pValues = &wait_fence_value,
                        };
                        vkWaitSemaphores(device_, &wait_info, ~0ULL);
                    }
                }
            }
        }

        this->ClearStallResources();
    }

    void VulkanSystem::GpuWait(GpuSystem::CmdQueueType type, uint64_t fence_value)
    {
        if (fence_value != GpuSystem::MaxFenceValue)
        {
            fence_val_ = std::max(fence_val_, fence_value);
        }

        const uint64_t curr_fence_value = fence_val_;
        ++fence_val_;

        VkQueue cmd_queue = this->GetOrCreateCommandQueue(type).cmd_queue;

        const VkTimelineSemaphoreSubmitInfo timeline_info{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &curr_fence_value,
            .signalSemaphoreValueCount = 0,
        };

        VkPipelineStageFlags wait_stage;
        switch (type)
        {
        case GpuSystem::CmdQueueType::Render:
            wait_stage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
            break;
        case GpuSystem::CmdQueueType::Compute:
            wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            break;
        case GpuSystem::CmdQueueType::VideoEncode:
            wait_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            break;
        }

        const VkSubmitInfo submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timeline_info,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &timeline_semaphore_,
            .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 0,
            .signalSemaphoreCount = 0,
        };

        TIFVK(vkQueueSubmit(cmd_queue, 1, &submit_info, VK_NULL_HANDLE));
    }

    uint64_t VulkanSystem::FenceValue() const noexcept
    {
        return fence_val_;
    }

    uint64_t VulkanSystem::CompletedFenceValue() const
    {
        uint64_t value;
        TIFVK(vkGetSemaphoreCounterValue(device_, timeline_semaphore_, &value));
        return value;
    }

    void VulkanSystem::HandleDeviceLost()
    {
        for (auto& cmd_queue : cmd_queues_)
        {
            cmd_queue.cmd_queue = VK_NULL_HANDLE;
            cmd_queue.cmd_pools.clear();
            cmd_queue.free_cmd_lists.clear();
        }

        vkDestroySemaphore(device_, timeline_semaphore_, nullptr);
        timeline_semaphore_ = VK_NULL_HANDLE;

        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    void VulkanSystem::Recycle(VkBuffer buff)
    {
        stall_resources_.emplace_back(buff, fence_val_, [this, buff]() { vkDestroyBuffer(device_, buff, nullptr); });
    }
    void VulkanSystem::Recycle(VkBufferView buff_view)
    {
        stall_resources_.emplace_back(buff_view, fence_val_, [this, buff_view]() { vkDestroyBufferView(device_, buff_view, nullptr); });
    }
    void VulkanSystem::Recycle(VkImage image)
    {
        stall_resources_.emplace_back(image, fence_val_, [this, image]() { vkDestroyImage(device_, image, nullptr); });
    }
    void VulkanSystem::Recycle(VkImageView image_view)
    {
        stall_resources_.emplace_back(image_view, fence_val_, [this, image_view]() { vkDestroyImageView(device_, image_view, nullptr); });
    }
    void VulkanSystem::Recycle(VkCommandPool cmd_pool)
    {
        stall_resources_.emplace_back(cmd_pool, fence_val_, [this, cmd_pool]() { vkDestroyCommandPool(device_, cmd_pool, nullptr); });
    }
    void VulkanSystem::Recycle(VkDescriptorPool desc_pool)
    {
        stall_resources_.emplace_back(desc_pool, fence_val_, [this, desc_pool]() { vkDestroyDescriptorPool(device_, desc_pool, nullptr); });
    }
    void VulkanSystem::Recycle(VkDescriptorSet desc_set, VkDescriptorPool desc_pool)
    {
        stall_resources_.emplace_back(
            desc_set, fence_val_, [this, desc_set, desc_pool]() { vkFreeDescriptorSets(device_, desc_pool, 1, &desc_set); });
    }
    void VulkanSystem::Recycle(VkSampler sampler)
    {
        stall_resources_.emplace_back(sampler, fence_val_, [this, sampler]() { vkDestroySampler(device_, sampler, nullptr); });
    }
    void VulkanSystem::Recycle(VkPipelineLayout layout)
    {
        stall_resources_.emplace_back(layout, fence_val_, [this, layout]() { vkDestroyPipelineLayout(device_, layout, nullptr); });
    }
    void VulkanSystem::Recycle(VkDescriptorSetLayout layout)
    {
        stall_resources_.emplace_back(layout, fence_val_, [this, layout]() { vkDestroyDescriptorSetLayout(device_, layout, nullptr); });
    }
    void VulkanSystem::Recycle(VkPipeline pipeline)
    {
        stall_resources_.emplace_back(pipeline, fence_val_, [this, pipeline]() { vkDestroyPipeline(device_, pipeline, nullptr); });
    }
    void VulkanSystem::Recycle(VkDeviceMemory memory)
    {
        stall_resources_.emplace_back(memory, fence_val_, [this, memory]() { vkFreeMemory(device_, memory, nullptr); });
    }
    void VulkanSystem::Recycle(VkRenderPass render_pass)
    {
        stall_resources_.emplace_back(
            render_pass, fence_val_, [this, render_pass]() { vkDestroyRenderPass(device_, render_pass, nullptr); });
    }

    void VulkanSystem::ClearStallResources()
    {
        const uint64_t completed_fence = this->CompletedFenceValue();
        for (auto iter = stall_resources_.begin(); iter != stall_resources_.end();)
        {
            if (iter->fence <= completed_fence)
            {
                iter = stall_resources_.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }

    VulkanSystem::CmdQueue& VulkanSystem::GetOrCreateCommandQueue(GpuSystem::CmdQueueType type)
    {
        auto& cmd_queue = cmd_queues_[static_cast<uint32_t>(type)];
        if (cmd_queue.cmd_queue == VK_NULL_HANDLE)
        {
            vkGetDeviceQueue(device_, this->QueueFamilyIndex(type), 0, &cmd_queue.cmd_queue);

            const std::string debug_name = std::format("cmd_queue {}", static_cast<uint32_t>(type));
            SetName(*this, cmd_queue.cmd_queue, VK_OBJECT_TYPE_QUEUE, debug_name);
        }

        return cmd_queue;
    }

    GpuCommandPool& VulkanSystem::CurrentCommandPool(GpuSystem::CmdQueueType type)
    {
        auto& cmd_queue = this->GetOrCreateCommandQueue(type);
        uint64_t completed_fence;
        TIFVK(vkGetSemaphoreCounterValue(device_, timeline_semaphore_, &completed_fence));
        for (auto& pool : cmd_queue.cmd_pools)
        {
            auto& vulkan_pool = VulkanImp(*pool);
            if (vulkan_pool.EmptyAllocatedCommandBuffers() && (vulkan_pool.FenceValue() <= completed_fence))
            {
                TIFVK(vkResetCommandPool(device_, vulkan_pool.CmdPool(), VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));
                return *pool;
            }
        }

        return *cmd_queue.cmd_pools.emplace_back(std::make_unique<GpuCommandPool>(*gpu_system_, type));
    }

    uint64_t VulkanSystem::ExecuteOnly(VulkanCommandList& cmd_list, uint64_t wait_fence_value)
    {
        auto& cmd_pool = *cmd_list.CommandPool();
        cmd_list.Close();

        const GpuSystem::CmdQueueType type = cmd_list.Type();
        VkQueue cmd_queue = this->GetOrCreateCommandQueue(type).cmd_queue;

        const uint64_t curr_fence_value = fence_val_;
        ++fence_val_;

        VkCommandBuffer cmd_buffs[] = {cmd_list.CommandBuffer()};

        VkTimelineSemaphoreSubmitInfo timeline_info{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &curr_fence_value,
        };

        VkSubmitInfo submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timeline_info,
            .commandBufferCount = static_cast<uint32_t>(std::size(cmd_buffs)),
            .pCommandBuffers = cmd_buffs,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &timeline_semaphore_,
        };

        if (wait_fence_value != GpuSystem::MaxFenceValue)
        {
            timeline_info.waitSemaphoreValueCount = 1;
            timeline_info.pWaitSemaphoreValues = &wait_fence_value;

            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = &timeline_semaphore_;

            VkPipelineStageFlags wait_stage;
            switch (type)
            {
            case GpuSystem::CmdQueueType::Render:
                wait_stage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
                break;
            case GpuSystem::CmdQueueType::Compute:
                wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                break;
            case GpuSystem::CmdQueueType::VideoEncode:
                wait_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                break;
            }
            submit_info.pWaitDstStageMask = &wait_stage;
        }

        TIFVK(vkQueueSubmit(cmd_queue, 1, &submit_info, VK_NULL_HANDLE));

        VulkanImp(cmd_pool).FenceValue(fence_val_);

        this->ClearStallResources();

        return curr_fence_value;
    }

    std::unique_ptr<GpuBufferInternal> VulkanSystem::CreateBuffer(
        uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name) const
    {
        return std::make_unique<VulkanBuffer>(*gpu_system_, size, heap, flags, std::move(name));
    }

    std::unique_ptr<GpuTextureInternal> VulkanSystem::CreateTexture(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
        uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::string_view name) const
    {
        return std::make_unique<VulkanTexture>(
            *gpu_system_, type, width, height, depth, array_size, mip_levels, format, flags, std::move(name));
    }

    std::unique_ptr<GpuStaticSamplerInternal> VulkanSystem::CreateStaticSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<VulkanStaticSampler>(*gpu_system_, filters, addr_modes);
    }

    std::unique_ptr<GpuDynamicSamplerInternal> VulkanSystem::CreateDynamicSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<VulkanDynamicSampler>(*gpu_system_, filters, addr_modes);
    }

    std::unique_ptr<GpuVertexLayoutInternal> VulkanSystem::CreateVertexLayout(
        std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides) const
    {
        return std::make_unique<VulkanVertexLayout>(std::move(attribs), std::move(slot_strides));
    }

    std::unique_ptr<GpuShaderResourceViewInternal> VulkanSystem::CreateShaderResourceView(
        const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanShaderResourceView>(*gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> VulkanSystem::CreateShaderResourceView(
        const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanShaderResourceView>(*gpu_system_, texture_array, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> VulkanSystem::CreateShaderResourceView(
        const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanShaderResourceView>(*gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> VulkanSystem::CreateShaderResourceView(
        const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const
    {
        return std::make_unique<VulkanShaderResourceView>(*gpu_system_, buffer, first_element, num_elements, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> VulkanSystem::CreateShaderResourceView(
        const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const
    {
        return std::make_unique<VulkanShaderResourceView>(*gpu_system_, buffer, first_element, num_elements, element_size);
    }

    std::unique_ptr<GpuRenderTargetViewInternal> VulkanSystem::CreateRenderTargetView(GpuTexture2D& texture, GpuFormat format) const
    {
        return std::make_unique<VulkanRenderTargetView>(*gpu_system_, texture, format);
    }

    std::unique_ptr<GpuDepthStencilViewInternal> VulkanSystem::CreateDepthStencilView(GpuTexture2D& texture, GpuFormat format) const
    {
        return std::make_unique<VulkanDepthStencilView>(*gpu_system_, texture, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> VulkanSystem::CreateUnorderedAccessView(
        GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanUnorderedAccessView>(*gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> VulkanSystem::CreateUnorderedAccessView(
        GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanUnorderedAccessView>(*gpu_system_, texture_array, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> VulkanSystem::CreateUnorderedAccessView(
        GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<VulkanUnorderedAccessView>(*gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> VulkanSystem::CreateUnorderedAccessView(
        GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const
    {
        return std::make_unique<VulkanUnorderedAccessView>(*gpu_system_, buffer, first_element, num_elements, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> VulkanSystem::CreateUnorderedAccessView(
        GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const
    {
        return std::make_unique<VulkanUnorderedAccessView>(*gpu_system_, buffer, first_element, num_elements, element_size);
    }

    std::unique_ptr<GpuRenderPipelineInternal> VulkanSystem::CreateRenderPipeline(GpuRenderPipeline::PrimitiveTopology topology,
        std::span<const ShaderInfo> shaders, const GpuVertexLayout& vertex_layout, std::span<const GpuStaticSampler> static_samplers,
        const GpuRenderPipeline::States& states) const
    {
        return std::make_unique<VulkanRenderPipeline>(
            *gpu_system_, topology, std::move(shaders), vertex_layout, std::move(static_samplers), states);
    }

    std::unique_ptr<GpuComputePipelineInternal> VulkanSystem::CreateComputePipeline(
        const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers) const
    {
        return std::make_unique<VulkanComputePipeline>(*gpu_system_, shader, std::move(static_samplers));
    }

    std::unique_ptr<GpuCommandPoolInternal> VulkanSystem::CreateCommandPool(GpuSystem::CmdQueueType type) const
    {
        return std::make_unique<VulkanCommandPool>(*gpu_system_, type);
    }

    std::unique_ptr<GpuCommandListInternal> VulkanSystem::CreateCommandList(GpuCommandPool& cmd_pool, GpuSystem::CmdQueueType type) const
    {
        return std::make_unique<VulkanCommandList>(*gpu_system_, cmd_pool, type);
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

    VkDescriptorSet VulkanSystem::AllocDescSet(VkDescriptorSetLayout layout)
    {
        if (!desc_set_allocators_)
        {
            desc_set_allocators_ = VulkanDescriptorSetAllocator(*gpu_system_);
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
