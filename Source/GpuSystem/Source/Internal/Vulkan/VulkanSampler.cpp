// Copyright (c) 2025 Minmin Gong
//

#include "VulkanSampler.hpp"

#include "Base/ErrorHandling.hpp"
#include "Gpu/GpuSystem.hpp"

#include "VulkanSystem.hpp"

namespace AIHoloImager
{
    void FillSamplerDesc(VkSamplerCreateInfo& sampler, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes)
    {
        sampler = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        };

        auto to_vulkan_min_mag_filter = [](const GpuSampler::Filter& filter) {
            switch (filter)
            {
            case GpuSampler::Filter::Point:
                return VK_FILTER_NEAREST;
            case GpuSampler::Filter::Linear:
                return VK_FILTER_LINEAR;
            default:
                Unreachable("Invalid filter");
            }
        };
        sampler.minFilter = to_vulkan_min_mag_filter(filters.min);
        sampler.magFilter = to_vulkan_min_mag_filter(filters.mag);

        auto to_vulkan_mip_filter = [](const GpuSampler::Filter& filter) {
            switch (filter)
            {
            case GpuSampler::Filter::Point:
                return VK_SAMPLER_MIPMAP_MODE_NEAREST;
            case GpuSampler::Filter::Linear:
                return VK_SAMPLER_MIPMAP_MODE_LINEAR;

            default:
                Unreachable("Invalid filter");
            }
        };
        sampler.mipmapMode = to_vulkan_mip_filter(filters.mip);

        auto to_vulkan_addr_mode = [](GpuSampler::AddressMode mode) {
            switch (mode)
            {
            case GpuSampler::AddressMode::Wrap:
                return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case GpuSampler::AddressMode::Mirror:
                return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            case GpuSampler::AddressMode::Clamp:
                return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            case GpuSampler::AddressMode::Border:
                return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            case GpuSampler::AddressMode::MirrorOnce:
                return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;

            default:
                Unreachable("Invalid address mode");
            }
        };
        sampler.addressModeU = to_vulkan_addr_mode(addr_modes.u);
        sampler.addressModeU = to_vulkan_addr_mode(addr_modes.v);
        sampler.addressModeU = to_vulkan_addr_mode(addr_modes.w);

        sampler.anisotropyEnable = false;
        sampler.compareEnable = false;
        sampler.minLod = 0.0f;
        sampler.maxLod = VK_LOD_CLAMP_NONE;
    }


    VULKAN_IMP_IMP(StaticSampler)

    VulkanStaticSampler::VulkanStaticSampler(
        GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes)
        : sampler_(std::make_shared<VulkanRecyclableObject<VkSampler>>(VulkanImp(gpu_system), VK_NULL_HANDLE))
    {
        const VkDevice vulkan_device = VulkanImp(gpu_system).Device();

        VkSamplerCreateInfo sampler_create_info;
        FillSamplerDesc(sampler_create_info, filters, addr_modes);
        vkCreateSampler(vulkan_device, &sampler_create_info, nullptr, &sampler_->Object());
    }

    VulkanStaticSampler::~VulkanStaticSampler() = default;

    VulkanStaticSampler::VulkanStaticSampler(VulkanStaticSampler&& other) noexcept = default;
    VulkanStaticSampler::VulkanStaticSampler(GpuStaticSamplerInternal&& other) noexcept
        : VulkanStaticSampler(static_cast<VulkanStaticSampler&&>(other))
    {
    }

    VulkanStaticSampler& VulkanStaticSampler::operator=(VulkanStaticSampler&& other) noexcept = default;
    GpuStaticSamplerInternal& VulkanStaticSampler::operator=(GpuStaticSamplerInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanStaticSampler&&>(other));
    }

    const std::shared_ptr<VulkanRecyclableObject<VkSampler>>& VulkanStaticSampler::Sampler() const noexcept
    {
        return sampler_;
    }


    VULKAN_IMP_IMP(DynamicSampler)

    VulkanDynamicSampler::VulkanDynamicSampler(
        GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes)
        : sampler_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = VulkanImp(gpu_system).Device();

        VkSamplerCreateInfo sampler_create_info;
        FillSamplerDesc(sampler_create_info, filters, addr_modes);
        vkCreateSampler(vulkan_device, &sampler_create_info, nullptr, &sampler_.Object());

        image_info_ = VkDescriptorImageInfo{
            .sampler = sampler_.Object(),
        };
        write_desc_set_ = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .pImageInfo = &image_info_,
        };
    }

    VulkanDynamicSampler::~VulkanDynamicSampler() = default;

    VulkanDynamicSampler::VulkanDynamicSampler(VulkanDynamicSampler&& other) noexcept = default;
    VulkanDynamicSampler::VulkanDynamicSampler(GpuDynamicSamplerInternal&& other) noexcept
        : VulkanDynamicSampler(static_cast<VulkanDynamicSampler&&>(other))
    {
    }

    VulkanDynamicSampler& VulkanDynamicSampler::operator=(VulkanDynamicSampler&& other) noexcept = default;
    GpuDynamicSamplerInternal& VulkanDynamicSampler::operator=(GpuDynamicSamplerInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanDynamicSampler&&>(other));
    }

    const VkSampler& VulkanDynamicSampler::Sampler() const noexcept
    {
        return sampler_.Object();
    }
    VkWriteDescriptorSet VulkanDynamicSampler::WriteDescSet() const noexcept
    {
        return write_desc_set_;
    }

    VkWriteDescriptorSet NullDynamicSamplerWriteDescSet()
    {
        static VkDescriptorImageInfo null_image_info{
            .sampler = VK_NULL_HANDLE,
        };
        static VkWriteDescriptorSet null_write_desc_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .pImageInfo = &null_image_info,
        };
        return null_write_desc_set;
    }
} // namespace AIHoloImager
