// Copyright (c) 2026 Minmin Gong
//

#include "VulkanQuery.hpp"

#include "Base/ErrorHandling.hpp"

#include "VulkanBuffer.hpp"
#include "VulkanCommandList.hpp"
#include "VulkanErrorhandling.hpp"
#include "VulkanSystem.hpp"

namespace AIHoloImager
{
    VulkanTimerQuery::VulkanTimerQuery(GpuSystem& gpu_system) : timestamp_pool_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        auto& vulkan_system = *timestamp_pool_.VulkanSys();
        const VkDevice vulkan_device = vulkan_system.Device();

        VkQueryPoolCreateInfo query_pool_info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
        };
        TIFVK(vkCreateQueryPool(vulkan_device, &query_pool_info, nullptr, &timestamp_pool_.Object()));
    }

    VulkanTimerQuery::~VulkanTimerQuery() = default;

    VulkanTimerQuery::VulkanTimerQuery(VulkanTimerQuery&& other) noexcept = default;
    VulkanTimerQuery::VulkanTimerQuery(GpuTimerQueryInternal&& other) noexcept : VulkanTimerQuery(static_cast<VulkanTimerQuery&&>(other))
    {
    }

    VulkanTimerQuery& VulkanTimerQuery::operator=(VulkanTimerQuery&& other) noexcept = default;
    VulkanTimerQuery& VulkanTimerQuery::operator=(GpuTimerQueryInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanTimerQuery&&>(other));
    }

    void VulkanTimerQuery::Begin(GpuCommandList& cmd_list)
    {
        auto& vulkan_cmd_list = VulkanImp(cmd_list);
        vkCmdResetQueryPool(vulkan_cmd_list.CommandBuffer(), timestamp_pool_.Object(), 0, 2);
        vkCmdWriteTimestamp(vulkan_cmd_list.CommandBuffer(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamp_pool_.Object(), 0);
    }

    void VulkanTimerQuery::End(GpuCommandList& cmd_list)
    {
        auto& vulkan_cmd_list = VulkanImp(cmd_list);
        vkCmdWriteTimestamp(vulkan_cmd_list.CommandBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamp_pool_.Object(), 1);
    }

    std::chrono::duration<double> VulkanTimerQuery::Elapsed() const
    {
        auto& vulkan_system = *timestamp_pool_.VulkanSys();
        const VkDevice vulkan_device = vulkan_system.Device();
        const float freq = vulkan_system.TimestampFrequency();

        uint64_t timestamps[2];
        vkGetQueryPoolResults(vulkan_device, timestamp_pool_.Object(), 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        const double seconds = static_cast<double>(timestamps[1] - timestamps[0]) * freq / 1000000000.0;
        return std::chrono::duration<double>(seconds);
    }
} // namespace AIHoloImager
