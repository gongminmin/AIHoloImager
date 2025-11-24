// Copyright (c) 2025 Minmin Gong
//

#include "VulkanUtil.hpp"

#include "Base/Util.hpp"

#include "VulkanSystem.hpp"

namespace AIHoloImager
{
    void SetName(const VulkanSystem& vulkan_system, const void* vulkan_object, VkObjectType type, std::string_view name)
    {
        const VkDevice vulkan_device = vulkan_system.Device();

        const std::string debug_name(std::move(name));
        const VkDebugUtilsObjectNameInfoEXT debug_name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType = type,
            .objectHandle = reinterpret_cast<uint64_t>(vulkan_object),
            .pObjectName = debug_name.c_str(),
        };
        vkSetDebugUtilsObjectNameEXT(vulkan_device, &debug_name_info);
    }
} // namespace AIHoloImager
