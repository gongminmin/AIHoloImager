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
        auto wait_semaphores = std::make_unique<VkSemaphore[]>(num_fences);
        auto wait_fence_values = std::make_unique<uint64_t[]>(num_fences);
        auto wait_stages = std::make_unique<VkPipelineStageFlags[]>(num_fences);

        for (uint32_t i = 0; i < num_fences; ++i)
        {
            const auto& [fence, value] = fences[i];
            wait_semaphores[i] = VulkanImp(*fence).Fence();
            wait_fence_values[i] = value;
            wait_stages[i] = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }

        const VkTimelineSemaphoreSubmitInfo timeline_info{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .waitSemaphoreValueCount = num_fences,
            .pWaitSemaphoreValues = wait_fence_values.get(),
            .signalSemaphoreValueCount = 0,
        };

        const VkSubmitInfo submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timeline_info,
            .waitSemaphoreCount = num_fences,
            .pWaitSemaphores = wait_semaphores.get(),
            .pWaitDstStageMask = wait_stages.get(),
            .commandBufferCount = 0,
            .signalSemaphoreCount = 0,
        };

        TIFVK(vkQueueSubmit(cmd_queue_.Object(), 1, &submit_info, VK_NULL_HANDLE));
    }

    void VulkanCommandQueue::Execute(const GpuCommandList& cmd_list, std::span<const GpuCommandQueue::FenceInfo> wait_fences,
        const GpuCommandQueue::FenceInfo& signal_fence)
    {
        this->Execute(VulkanImp(cmd_list), std::move(wait_fences), signal_fence);
    }
    void VulkanCommandQueue::Execute(const VulkanCommandList& cmd_list, std::span<const GpuCommandQueue::FenceInfo> wait_fences,
        const GpuCommandQueue::FenceInfo& signal_fence)
    {
        const VkCommandBuffer vulkan_cmd_buffs[] = {cmd_list.CommandBuffer()};
        const VkSemaphore semaphores[] = {VulkanImp(*signal_fence.fence).Fence()};

        VkTimelineSemaphoreSubmitInfo timeline_info{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &signal_fence.value,
        };

        VkSubmitInfo submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timeline_info,
            .commandBufferCount = static_cast<uint32_t>(std::size(vulkan_cmd_buffs)),
            .pCommandBuffers = vulkan_cmd_buffs,
            .signalSemaphoreCount = static_cast<uint32_t>(std::size(semaphores)),
            .pSignalSemaphores = semaphores,
        };

        if (!wait_fences.empty())
        {
            const uint32_t num_fences = static_cast<uint32_t>(wait_fences.size());
            auto wait_semaphores = std::make_unique<VkSemaphore[]>(num_fences);
            auto wait_fence_values = std::make_unique<uint64_t[]>(num_fences);
            auto wait_stages = std::make_unique<VkPipelineStageFlags[]>(num_fences);

            for (uint32_t i = 0; i < num_fences; ++i)
            {
                const auto& [fence, value] = wait_fences[i];
                wait_semaphores[i] = VulkanImp(*fence).Fence();
                wait_fence_values[i] = value;
                wait_stages[i] = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            }

            timeline_info.waitSemaphoreValueCount = num_fences;
            timeline_info.pWaitSemaphoreValues = wait_fence_values.get();

            submit_info.waitSemaphoreCount = num_fences;
            submit_info.pWaitSemaphores = wait_semaphores.get();
            submit_info.pWaitDstStageMask = wait_stages.get();
        }

        TIFVK(vkQueueSubmit(cmd_queue_.Object(), 1, &submit_info, VK_NULL_HANDLE));
    }

    VkQueue VulkanCommandQueue::CmdQueue() const noexcept
    {
        return cmd_queue_.Object();
    }
} // namespace AIHoloImager
