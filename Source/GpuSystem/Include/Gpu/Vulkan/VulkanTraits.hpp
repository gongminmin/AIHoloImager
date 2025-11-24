// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <volk.h>

namespace AIHoloImager
{
    struct VulkanTraits
    {
        using DeviceType = VkDevice;
        using CommandQueueType = VkQueue;

        using CommandListType = VkCommandBuffer;
        using GraphicsCommandListType = VkCommandBuffer;
        using ComputeCommandListType = VkCommandBuffer;
        using VideoEncodeCommandListType = VkCommandBuffer;

        using ResourceType = VkDeviceMemory;
        using BufferType = VkBuffer;
        using TextureType = VkImage;

        using SharedHandleType = HANDLE;
    };
} // namespace AIHoloImager
