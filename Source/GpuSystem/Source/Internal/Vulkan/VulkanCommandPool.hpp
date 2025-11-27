// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <list>

#include <volk.h>

#include "Gpu/GpuCommandPool.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandPoolInternal.hpp"
#include "VulkanImpDefine.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class VulkanCommandPool : public GpuCommandPoolInternal
    {
    public:
        VulkanCommandPool(GpuSystem& gpu_system, GpuSystem::CmdQueueType type);
        ~VulkanCommandPool() noexcept override;

        VulkanCommandPool(VulkanCommandPool&& other) noexcept;
        explicit VulkanCommandPool(GpuCommandPoolInternal&& other) noexcept;
        VulkanCommandPool& operator=(VulkanCommandPool&& other) noexcept;
        GpuCommandPoolInternal& operator=(GpuCommandPoolInternal&& other) noexcept override;

        VkCommandPool CmdPool() const noexcept;

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

    VULKAN_DEFINE_IMP(CommandPool)
} // namespace AIHoloImager
