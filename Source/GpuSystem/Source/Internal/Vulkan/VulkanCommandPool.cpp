// Copyright (c) 2025 Minmin Gong
//

#include "VulkanCommandPool.hpp"

#include "Base/ErrorHandling.hpp"

#include "VulkanErrorhandling.hpp"
#include "VulkanSystem.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(CommandPool)

    VulkanCommandPool::VulkanCommandPool(GpuSystem& gpu_system, GpuSystem::CmdQueueType type)
        : cmd_pool_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        auto& vulkan_system = *cmd_pool_.VulkanSys();
        const VkDevice device = vulkan_system.Device();

        const VkCommandPoolCreateInfo cmd_pool_create_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = 0,
            .queueFamilyIndex = vulkan_system.QueueFamilyIndex(type),
        };
        TIFVK(vkCreateCommandPool(device, &cmd_pool_create_info, nullptr, &cmd_pool_.Object()));
    }

    VulkanCommandPool::~VulkanCommandPool() noexcept = default;

    VulkanCommandPool::VulkanCommandPool(VulkanCommandPool&& other) noexcept = default;
    VulkanCommandPool::VulkanCommandPool(GpuCommandPoolInternal&& other) noexcept
        : VulkanCommandPool(static_cast<VulkanCommandPool&&>(other))
    {
    }
    VulkanCommandPool& VulkanCommandPool::operator=(VulkanCommandPool&& other) noexcept = default;
    GpuCommandPoolInternal& VulkanCommandPool::operator=(GpuCommandPoolInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanCommandPool&&>(other));
    }

    VkCommandPool VulkanCommandPool::CmdAllocator() const noexcept
    {
        return cmd_pool_.Object();
    }

    uint64_t VulkanCommandPool::FenceValue() const noexcept
    {
        return fence_val_;
    }

    void VulkanCommandPool::FenceValue(uint64_t value) noexcept
    {
        fence_val_ = value;
    }

    void VulkanCommandPool::RegisterAllocatedCommandBuffer(VkCommandBuffer cmd_buff)
    {
        allocated_cmd_buffs_.push_back(cmd_buff);
    }

    void VulkanCommandPool::UnregisterAllocatedCommandBuffer(VkCommandBuffer cmd_buff)
    {
        auto iter = std::find(allocated_cmd_buffs_.begin(), allocated_cmd_buffs_.end(), cmd_buff);
        if (iter != allocated_cmd_buffs_.end())
        {
            allocated_cmd_buffs_.erase(iter);
        }
    }

    bool VulkanCommandPool::EmptyAllocatedCommandBuffers() const noexcept
    {
        return allocated_cmd_buffs_.empty();
    }
} // namespace AIHoloImager
