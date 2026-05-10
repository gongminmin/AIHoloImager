// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <volk.h>

#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuFence.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuFenceInternal.hpp"
#include "VulkanImpDefine.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class VulkanFence : public GpuFenceInternal
    {
        DISALLOW_COPY_AND_ASSIGN(VulkanFence)

    public:
        VulkanFence() noexcept;
        VulkanFence(GpuSystem& gpu_system, uint64_t init_val, bool enable_sharing);
        ~VulkanFence() noexcept override;

        VulkanFence(VulkanFence&& other) noexcept;
        explicit VulkanFence(GpuFenceInternal&& other) noexcept;
        VulkanFence& operator=(VulkanFence&& other) noexcept;
        GpuFenceInternal& operator=(GpuFenceInternal&& other) noexcept override;

        void* NativeFence() const noexcept;
        void* SharedFenceHandle() const noexcept override;

        uint64_t CompletedValue() const override;

        void CpuWait(uint64_t value) const override;

        VkSemaphore Fence() const noexcept;

    private:
        VulkanRecyclableObject<VkSemaphore> timeline_semaphore_;
        Win32UniqueHandle shared_fence_handle_;
    };

    VULKAN_DEFINE_IMP(Fence)
} // namespace AIHoloImager
