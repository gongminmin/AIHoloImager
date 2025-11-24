// Copyright (c) 2025 Minmin Gong
//

#include "VulkanDescriptorAllocator.hpp"

#include "VulkanErrorhandling.hpp"
#include "VulkanSystem.hpp"

namespace
{
    constexpr uint32_t DescriptorPageSize[] = {4 * 1024, 1 * 1024};
}

namespace AIHoloImager
{
    VulkanDescriptorPool::VulkanDescriptorPool(GpuSystem& gpu_system) : pool_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = pool_.VulkanSys()->Device();

        std::vector<VkDescriptorPoolSize> pool_count;
        for (auto type : {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC})
        {
            pool_count.emplace_back(VkDescriptorPoolSize{
                .type = type,
                .descriptorCount = DescriptorPageSize[0],
            });
        }
        pool_count.emplace_back(VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = DescriptorPageSize[1],
        });

        const VkDescriptorPoolCreateInfo descriptor_pool_create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 2,
            .poolSizeCount = static_cast<uint32_t>(pool_count.size()),
            .pPoolSizes = pool_count.data(),
        };
        TIFVK(vkCreateDescriptorPool(vulkan_device, &descriptor_pool_create_info, nullptr, &pool_.Object()));
    }

    VulkanDescriptorPool::~VulkanDescriptorPool() = default;

    VulkanDescriptorPool::VulkanDescriptorPool(VulkanDescriptorPool&& other) noexcept = default;
    VulkanDescriptorPool& VulkanDescriptorPool::operator=(VulkanDescriptorPool&& other) noexcept = default;

    VkDescriptorPool VulkanDescriptorPool::Pool() const noexcept
    {
        return pool_.Object();
    }


    VulkanDescriptorSetAllocator::VulkanDescriptorSetAllocator() = default;
    VulkanDescriptorSetAllocator::VulkanDescriptorSetAllocator(GpuSystem& gpu_system) : gpu_system_(&gpu_system)
    {
    }

    VulkanDescriptorSetAllocator::~VulkanDescriptorSetAllocator()
    {
    }

    VulkanDescriptorSetAllocator::VulkanDescriptorSetAllocator(VulkanDescriptorSetAllocator&& other) noexcept = default;
    VulkanDescriptorSetAllocator& VulkanDescriptorSetAllocator::operator=(VulkanDescriptorSetAllocator&& other) noexcept = default;

    VulkanDescriptorSetAllocator::operator bool() const
    {
        return gpu_system_ != nullptr;
    }

    VkDescriptorSet VulkanDescriptorSetAllocator::Allocate(VkDescriptorSetLayout layout)
    {
        const VkDevice vulkan_device = VulkanImp(*gpu_system_).Device();

        VkDescriptorSetAllocateInfo allocate_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };

        VulkanRecyclableObject<VkDescriptorSet> desc_set(VulkanImp(*gpu_system_), VK_NULL_HANDLE);
        for (auto& pool : pools_)
        {
            allocate_info.descriptorPool = pool.pool.Pool();
            VkResult result = vkAllocateDescriptorSets(vulkan_device, &allocate_info, &desc_set.Object());
            if (result == VK_SUCCESS)
            {
                desc_set.AddExtraRecycleParam(allocate_info.descriptorPool);
                return pool.allocated_sets.emplace_back(std::move(desc_set)).Object();
            }
        }

        auto& new_pool = pools_.emplace_back(PoolInfo{VulkanDescriptorPool(*gpu_system_), {}});
        allocate_info.descriptorPool = new_pool.pool.Pool();
        TIFVK(vkAllocateDescriptorSets(vulkan_device, &allocate_info, &desc_set.Object()));
        desc_set.AddExtraRecycleParam(allocate_info.descriptorPool);
        return new_pool.allocated_sets.emplace_back(std::move(desc_set)).Object();
    }

    void VulkanDescriptorSetAllocator::Deallocate(VkDescriptorSet desc_set)
    {
        for (auto& pool : pools_)
        {
            auto iter = std::find_if(pool.allocated_sets.begin(), pool.allocated_sets.end(),
                [desc_set](const VulkanRecyclableObject<VkDescriptorSet>& item) { return item.Object() == desc_set; });
            if (iter != pool.allocated_sets.end())
            {
                pool.allocated_sets.erase(iter);
                return;
            }
        }
    }
} // namespace AIHoloImager
