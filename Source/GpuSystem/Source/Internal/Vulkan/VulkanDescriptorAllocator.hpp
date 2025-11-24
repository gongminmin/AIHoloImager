// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <vector>

#include <volk.h>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuSystem.hpp"

#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class VulkanDescriptorPool
    {
        DISALLOW_COPY_AND_ASSIGN(VulkanDescriptorPool)

    public:
        explicit VulkanDescriptorPool(GpuSystem& gpu_system);
        ~VulkanDescriptorPool();

        VulkanDescriptorPool(VulkanDescriptorPool&& other) noexcept;
        VulkanDescriptorPool& operator=(VulkanDescriptorPool&& other) noexcept;

        VkDescriptorPool Pool() const noexcept;

    private:
        VulkanRecyclableObject<VkDescriptorPool> pool_;
    };

    class VulkanDescriptorSetAllocator final
    {
        DISALLOW_COPY_AND_ASSIGN(VulkanDescriptorSetAllocator)

    public:
        VulkanDescriptorSetAllocator();
        explicit VulkanDescriptorSetAllocator(GpuSystem& gpu_system);
        ~VulkanDescriptorSetAllocator();

        VulkanDescriptorSetAllocator(VulkanDescriptorSetAllocator&& other) noexcept;
        VulkanDescriptorSetAllocator& operator=(VulkanDescriptorSetAllocator&& other) noexcept;

        explicit operator bool() const;

        VkDescriptorSet Allocate(VkDescriptorSetLayout layout);
        void Deallocate(VkDescriptorSet desc_set);

    private:
        GpuSystem* gpu_system_ = nullptr;

        struct PoolInfo
        {
            VulkanDescriptorPool pool;
            std::vector<VulkanRecyclableObject<VkDescriptorSet>> allocated_sets;
        };

        std::vector<PoolInfo> pools_;
    };
} // namespace AIHoloImager
