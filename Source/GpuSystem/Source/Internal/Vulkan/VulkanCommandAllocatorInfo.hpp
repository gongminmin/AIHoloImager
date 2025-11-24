// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <list>

#include <volk.h>

#include "Gpu/GpuCommandAllocatorInfo.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandAllocatorInfoInternal.hpp"
#include "VulkanImpDefine.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class VulkanCommandAllocatorInfo : public GpuCommandAllocatorInfoInternal
    {
    public:
        VulkanCommandAllocatorInfo(GpuSystem& gpu_system, GpuSystem::CmdQueueType type);
        ~VulkanCommandAllocatorInfo() noexcept override;

        VulkanCommandAllocatorInfo(VulkanCommandAllocatorInfo&& other) noexcept;
        explicit VulkanCommandAllocatorInfo(GpuCommandAllocatorInfoInternal&& other) noexcept;
        VulkanCommandAllocatorInfo& operator=(VulkanCommandAllocatorInfo&& other) noexcept;
        GpuCommandAllocatorInfoInternal& operator=(GpuCommandAllocatorInfoInternal&& other) noexcept override;

        VkCommandPool CmdAllocator() const noexcept;

        uint64_t FenceValue() const noexcept;
        void FenceValue(uint64_t value) noexcept;

        void RegisterAllocatedCommandBuffer(VkCommandBuffer cmd_buff);
        void UnregisterAllocatedCommandBuffer(VkCommandBuffer cmd_buff);
        bool EmptyAllocatedCommandBuffers() const noexcept;

    private:
        VulkanRecyclableObject<VkCommandPool> cmd_pool_;
        uint64_t fence_val_ = 0;

        std::list<VkCommandBuffer> allocated_cmd_buffs_;
    };

    VULKAN_DEFINE_IMP(CommandAllocatorInfo)
} // namespace AIHoloImager
