// Copyright (c) 2025 Minmin Gong
//

#include "VulkanCommandAllocatorInfo.hpp"

#include "Base/ErrorHandling.hpp"

#include "VulkanErrorhandling.hpp"
#include "VulkanSystem.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(CommandAllocatorInfo)

    VulkanCommandAllocatorInfo::VulkanCommandAllocatorInfo(GpuSystem& gpu_system, GpuSystem::CmdQueueType type)
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

    VulkanCommandAllocatorInfo::~VulkanCommandAllocatorInfo() noexcept = default;

    VulkanCommandAllocatorInfo::VulkanCommandAllocatorInfo(VulkanCommandAllocatorInfo&& other) noexcept = default;
    VulkanCommandAllocatorInfo::VulkanCommandAllocatorInfo(GpuCommandAllocatorInfoInternal&& other) noexcept
        : VulkanCommandAllocatorInfo(static_cast<VulkanCommandAllocatorInfo&&>(other))
    {
    }
    VulkanCommandAllocatorInfo& VulkanCommandAllocatorInfo::operator=(VulkanCommandAllocatorInfo&& other) noexcept = default;
    GpuCommandAllocatorInfoInternal& VulkanCommandAllocatorInfo::operator=(GpuCommandAllocatorInfoInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanCommandAllocatorInfo&&>(other));
    }

    VkCommandPool VulkanCommandAllocatorInfo::CmdAllocator() const noexcept
    {
        return cmd_pool_.Object();
    }

    uint64_t VulkanCommandAllocatorInfo::FenceValue() const noexcept
    {
        return fence_val_;
    }

    void VulkanCommandAllocatorInfo::FenceValue(uint64_t value) noexcept
    {
        fence_val_ = value;
    }

    void VulkanCommandAllocatorInfo::RegisterAllocatedCommandBuffer(VkCommandBuffer cmd_buff)
    {
        allocated_cmd_buffs_.push_back(cmd_buff);
    }

    void VulkanCommandAllocatorInfo::UnregisterAllocatedCommandBuffer(VkCommandBuffer cmd_buff)
    {
        auto iter = std::find(allocated_cmd_buffs_.begin(), allocated_cmd_buffs_.end(), cmd_buff);
        if (iter != allocated_cmd_buffs_.end())
        {
            allocated_cmd_buffs_.erase(iter);
        }
    }

    bool VulkanCommandAllocatorInfo::EmptyAllocatedCommandBuffers() const noexcept
    {
        return allocated_cmd_buffs_.empty();
    }
} // namespace AIHoloImager
