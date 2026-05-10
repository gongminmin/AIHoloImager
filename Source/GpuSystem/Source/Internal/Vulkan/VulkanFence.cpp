// Copyright (c) 2026 Minmin Gong
//

#include "VulkanFence.hpp"

#include "VulkanErrorHandling.hpp"
#include "VulkanSystem.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(Fence)

    VulkanFence::VulkanFence() noexcept = default;
    VulkanFence::VulkanFence(GpuSystem& gpu_system, uint64_t init_val, bool enable_sharing)
        : timeline_semaphore_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const auto& vulkan_system = *timeline_semaphore_.VulkanSys();
        const VkDevice vulkan_device = vulkan_system.Device();

        VkSemaphoreTypeCreateInfo timeline_create_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = init_val,
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
        TIFVK(vkCreateSemaphore(vulkan_device, &semaphore_create_info, nullptr, &timeline_semaphore_.Object()));

        if (enable_sharing)
        {
            const VkSemaphoreGetWin32HandleInfoKHR get_handle_info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
                .semaphore = timeline_semaphore_.Object(),
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };

            HANDLE shared_handle;
            TIFVK(vkGetSemaphoreWin32HandleKHR(vulkan_device, &get_handle_info, &shared_handle));
            shared_fence_handle_.reset(shared_handle);
        }
    }

    VulkanFence::~VulkanFence() noexcept = default;

    VulkanFence::VulkanFence(VulkanFence&& other) noexcept = default;
    VulkanFence::VulkanFence(GpuFenceInternal&& other) noexcept : VulkanFence(static_cast<VulkanFence&&>(other))
    {
    }

    VulkanFence& VulkanFence::operator=(VulkanFence&& other) noexcept = default;
    GpuFenceInternal& VulkanFence::operator=(GpuFenceInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanFence&&>(other));
    }

    void* VulkanFence::NativeFence() const noexcept
    {
        return this->Fence();
    }

    void* VulkanFence::SharedFenceHandle() const noexcept
    {
        return shared_fence_handle_.get();
    }

    uint64_t VulkanFence::CompletedValue() const
    {
        if (!timeline_semaphore_)
        {
            return 0;
        }

        const auto& vulkan_system = *timeline_semaphore_.VulkanSys();
        const VkDevice vulkan_device = vulkan_system.Device();

        uint64_t completed_value;
        if (vkGetSemaphoreCounterValue(vulkan_device, timeline_semaphore_.Object(), &completed_value) == VK_SUCCESS)
        {
            return completed_value;
        }
        else
        {
            return 0;
        }
    }

    void VulkanFence::CpuWait(uint64_t value) const
    {
        const auto& vulkan_system = *timeline_semaphore_.VulkanSys();
        const VkDevice vulkan_device = vulkan_system.Device();

        const VkSemaphoreWaitInfo wait_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &timeline_semaphore_.Object(),
            .pValues = &value,
        };
        vkWaitSemaphores(vulkan_device, &wait_info, ~0ULL);
    }

    VkSemaphore VulkanFence::Fence() const noexcept
    {
        return timeline_semaphore_.Object();
    }
} // namespace AIHoloImager
