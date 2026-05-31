// Copyright (c) 2026 Minmin Gong
//

#include "VulkanCommandQueue.hpp"

#include "VulkanErrorHandling.hpp"
#include "VulkanFence.hpp"
#include "VulkanSystem.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(CommandQueue)

    VulkanCommandQueue::VulkanCommandQueue() noexcept = default;
    VulkanCommandQueue::VulkanCommandQueue(GpuSystem& gpu_system, GpuSystem::CmdQueueType type, std::string_view name)
        : cmd_queue_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        auto& vulkan_system = *cmd_queue_.VulkanSys();
        const VkDevice device = vulkan_system.Device();

        vkGetDeviceQueue(device, vulkan_system.QueueFamilyIndex(type), 0, &cmd_queue_.Object());
        SetName(vulkan_system, cmd_queue_.Object(), VK_OBJECT_TYPE_QUEUE, name);
    }

    VulkanCommandQueue::~VulkanCommandQueue() noexcept = default;

    VulkanCommandQueue::VulkanCommandQueue(VulkanCommandQueue&& other) noexcept = default;
    VulkanCommandQueue::VulkanCommandQueue(GpuCommandQueueInternal&& other) noexcept
        : VulkanCommandQueue(static_cast<VulkanCommandQueue&&>(other))
    {
    }
    VulkanCommandQueue& VulkanCommandQueue::operator=(VulkanCommandQueue&& other) noexcept = default;
    GpuCommandQueueInternal& VulkanCommandQueue::operator=(GpuCommandQueueInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanCommandQueue&&>(other));
    }

    VulkanCommandQueue::operator bool() const noexcept
    {
        return static_cast<bool>(cmd_queue_);
    }

    void* VulkanCommandQueue::NativeCmdQueue() const noexcept
    {
        return this->CmdQueue();
    }

    void VulkanCommandQueue::GpuWait(std::span<const GpuCommandQueue::FenceInfo> fences)
    {
        const uint32_t num_fences = static_cast<uint32_t>(fences.size());
        auto wait_semaphores = std::make_unique<VkSemaphoreSubmitInfo[]>(num_fences);
        for (uint32_t i = 0; i < num_fences; ++i)
        {
            const auto& [fence, value] = fences[i];
            wait_semaphores[i] = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .semaphore = VulkanImp(*fence).Fence(),
                .value = value,
                .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            };
        }

        const VkSubmitInfo2 submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .waitSemaphoreInfoCount = num_fences,
            .pWaitSemaphoreInfos = wait_semaphores.get(),
            .commandBufferInfoCount = 0,
            .signalSemaphoreInfoCount = 0,
        };

        TIFVK(vkQueueSubmit2(cmd_queue_.Object(), 1, &submit_info, VK_NULL_HANDLE));
    }

    void VulkanCommandQueue::Execute(const GpuCommandList& cmd_list, std::span<const GpuCommandQueue::FenceInfo> wait_fences,
        const GpuCommandQueue::FenceInfo& signal_fence)
    {
        this->Execute(cmd_list.Internal(), std::move(wait_fences), signal_fence);
    }
    void VulkanCommandQueue::Execute(const GpuCommandListInternal& cmd_list_internal,
        std::span<const GpuCommandQueue::FenceInfo> wait_fences, const GpuCommandQueue::FenceInfo& signal_fence)
    {
        const uint32_t num_fences = static_cast<uint32_t>(wait_fences.size());
        std::unique_ptr<VkSemaphoreSubmitInfo[]> wait_semaphores;
        if (num_fences > 0)
        {
            wait_semaphores = std::make_unique<VkSemaphoreSubmitInfo[]>(num_fences);
            for (uint32_t i = 0; i < num_fences; ++i)
            {
                const auto& [fence, value] = wait_fences[i];
                wait_semaphores[i] = {
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .semaphore = VulkanImp(*fence).Fence(),
                    .value = value,
                    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                };
            }
        }

        const VkCommandBufferSubmitInfo cmd_buff_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = VulkanImp(cmd_list_internal).CommandBuffer(),
        };

        const VkSemaphoreSubmitInfo signal_semaphore_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = VulkanImp(*signal_fence.fence).Fence(),
            .value = signal_fence.value,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };

        const VkSubmitInfo2 submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .waitSemaphoreInfoCount = num_fences,
            .pWaitSemaphoreInfos = wait_semaphores.get(),
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmd_buff_info,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos = &signal_semaphore_info,
        };

        TIFVK(vkQueueSubmit2(cmd_queue_.Object(), 1, &submit_info, VK_NULL_HANDLE));
    }

    VkQueue VulkanCommandQueue::CmdQueue() const noexcept
    {
        return cmd_queue_.Object();
    }
} // namespace AIHoloImager
