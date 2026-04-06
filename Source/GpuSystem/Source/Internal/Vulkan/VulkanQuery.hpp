// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <volk.h>

#include "Gpu/GpuQuery.hpp"

#include "../GpuQueryInternal.hpp"
#include "VulkanImpDefine.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class VulkanTimerQuery : public GpuTimerQueryInternal
    {
    public:
        explicit VulkanTimerQuery(GpuSystem& gpu_system);
        ~VulkanTimerQuery() override;

        VulkanTimerQuery(VulkanTimerQuery&& other) noexcept;
        explicit VulkanTimerQuery(GpuTimerQueryInternal&& other) noexcept;

        VulkanTimerQuery& operator=(VulkanTimerQuery&& other) noexcept;
        VulkanTimerQuery& operator=(GpuTimerQueryInternal&& other) noexcept override;

        void Begin(GpuCommandList& cmd_list) override;
        void End(GpuCommandList& cmd_list) override;

        std::chrono::duration<double> Elapsed() const override;

    private:
        VulkanRecyclableObject<VkQueryPool> timestamp_pool_;
    };

    VULKAN_DEFINE_IMP(TimerQuery)
} // namespace AIHoloImager
