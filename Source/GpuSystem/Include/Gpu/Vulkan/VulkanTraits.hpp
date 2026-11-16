// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include <volk.h>

#include "Gpu/GpuSystem.hpp"

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

        using FenceType = VkSemaphore;

        using SharedHandleType = HANDLE;

        struct DeviceInfo : public GpuSystem::DeviceInfo
        {
            VkInstance instance;
            VkPhysicalDevice physical_device;
        };
    };
} // namespace AIHoloImager
