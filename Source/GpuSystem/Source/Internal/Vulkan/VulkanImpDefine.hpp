// Copyright (c) 2025 Minmin Gong
//

#pragma once

#define VULKAN_DEFINE_IMP(ClassName)                   \
    Vulkan##ClassName& VulkanImp(Gpu##ClassName& var); \
    const Vulkan##ClassName& VulkanImp(const Gpu##ClassName& var);

#define VULKAN_IMP_IMP(ClassName)                                     \
    Vulkan##ClassName& VulkanImp(Gpu##ClassName& var)                 \
    {                                                                 \
        return static_cast<Vulkan##ClassName&>(var.Internal());       \
    }                                                                 \
                                                                      \
    const Vulkan##ClassName& VulkanImp(const Gpu##ClassName& var)     \
    {                                                                 \
        return static_cast<const Vulkan##ClassName&>(var.Internal()); \
    }
