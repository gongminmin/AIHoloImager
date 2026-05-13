// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <volk.h>

#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandQueueInternal.hpp"
#include "VulkanCommandList.hpp"
#include "VulkanImpDefine.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class VulkanCommandQueue : public GpuCommandQueueInternal
    {
    public:
        VulkanCommandQueue() noexcept;
        VulkanCommandQueue(GpuSystem& gpu_system, GpuSystem::CmdQueueType type, std::string_view name);
        ~VulkanCommandQueue() noexcept override;

        VulkanCommandQueue(VulkanCommandQueue&& other) noexcept;
        explicit VulkanCommandQueue(GpuCommandQueueInternal&& other) noexcept;
        VulkanCommandQueue& operator=(VulkanCommandQueue&& other) noexcept;
        GpuCommandQueueInternal& operator=(GpuCommandQueueInternal&& other) noexcept;

        explicit operator bool() const noexcept override;

        void* NativeCmdQueue() const noexcept override;

        void GpuWait(std::span<const GpuCommandQueue::FenceInfo> fences) override;

        void Execute(const GpuCommandList& cmd_list, std::span<const GpuCommandQueue::FenceInfo> wait_fences,
            const GpuCommandQueue::FenceInfo& signal_fence) override;
        void Execute(const GpuCommandListInternal& cmd_list_internal, std::span<const GpuCommandQueue::FenceInfo> wait_fences,
            const GpuCommandQueue::FenceInfo& signal_fence) override;

        VkQueue CmdQueue() const noexcept;

    private:
        VulkanRecyclableObject<VkQueue> cmd_queue_;
    };

    VULKAN_DEFINE_IMP(CommandQueue)
} // namespace AIHoloImager
