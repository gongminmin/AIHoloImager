// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

#include <volk.h>

#include "Gpu/GpuSampler.hpp"

#include "../GpuSamplerInternal.hpp"
#include "VulkanImpDefine.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class VulkanStaticSampler : public GpuStaticSamplerInternal
    {
    public:
        VulkanStaticSampler(GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes);
        ~VulkanStaticSampler() override;

        VulkanStaticSampler(VulkanStaticSampler&& other) noexcept;
        explicit VulkanStaticSampler(GpuStaticSamplerInternal&& other) noexcept;

        VulkanStaticSampler& operator=(VulkanStaticSampler&& other) noexcept;
        GpuStaticSamplerInternal& operator=(GpuStaticSamplerInternal&& other) noexcept override;

        const std::shared_ptr<VulkanRecyclableObject<VkSampler>>& Sampler() const noexcept;

    private:
        std::shared_ptr<VulkanRecyclableObject<VkSampler>> sampler_;
    };

    VULKAN_DEFINE_IMP(StaticSampler)

    class VulkanDynamicSampler : public GpuDynamicSamplerInternal
    {
    public:
        VulkanDynamicSampler(GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes);
        ~VulkanDynamicSampler() override;

        VulkanDynamicSampler(VulkanDynamicSampler&& other) noexcept;
        explicit VulkanDynamicSampler(GpuDynamicSamplerInternal&& other) noexcept;

        VulkanDynamicSampler& operator=(VulkanDynamicSampler&& other) noexcept;
        GpuDynamicSamplerInternal& operator=(GpuDynamicSamplerInternal&& other) noexcept override;

        const VkSampler& Sampler() const noexcept;
        VkWriteDescriptorSet WriteDescSet() const noexcept;

    private:
        VulkanRecyclableObject<VkSampler> sampler_;
        VkWriteDescriptorSet write_desc_set_;
        VkDescriptorImageInfo image_info_;
    };

    VULKAN_DEFINE_IMP(DynamicSampler)

    VkWriteDescriptorSet NullDynamicSamplerWriteDescSet();
} // namespace AIHoloImager
