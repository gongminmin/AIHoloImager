// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <tuple>

#include <volk.h>

#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResource.hpp"

namespace AIHoloImager
{
    VkFormat ToVkFormat(GpuFormat fmt);
    GpuFormat FromVkFormat(VkFormat fmt);

    VkMemoryPropertyFlags ToVulkanMemoryPropertyFlags(GpuHeap heap);
    GpuHeap FromVulkanMemoryPropertyFlags(VkMemoryPropertyFlags flags);

    VkImageUsageFlags ToVulkanImageUsageFlags(GpuResourceFlag flags) noexcept;
    GpuResourceFlag FromVulkanImageUsageFlags(VkImageUsageFlags flags) noexcept;

    VkImageLayout ToVulkanImageLayout(GpuResourceState state);

    VkImageType ToVulkanImageType(GpuResourceType type);

    std::tuple<VkAccessFlags, VkAccessFlags> ToVulkanAccessFlags(VkImageLayout old_layout, VkImageLayout new_layout);

    VkImageAspectFlags ToVulkanAspectMask(GpuFormat fmt) noexcept;
} // namespace AIHoloImager
