// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <span>
#include <vector>

#include <volk.h>

#include "Gpu/GpuVertexAttrib.hpp"

#include "../GpuVertexAttribInternal.hpp"
#include "VulkanImpDefine.hpp"

namespace AIHoloImager
{
    class VulkanVertexAttribs : public GpuVertexAttribsInternal
    {
    public:
        explicit VulkanVertexAttribs(std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides = {});
        ~VulkanVertexAttribs() override;

        VulkanVertexAttribs(const VulkanVertexAttribs& other);
        explicit VulkanVertexAttribs(const GpuVertexAttribsInternal& other);

        VulkanVertexAttribs& operator=(const VulkanVertexAttribs& other);
        GpuVertexAttribsInternal& operator=(const GpuVertexAttribsInternal& other) override;

        VulkanVertexAttribs(VulkanVertexAttribs&& other) noexcept;
        explicit VulkanVertexAttribs(GpuVertexAttribsInternal&& other) noexcept;

        VulkanVertexAttribs& operator=(VulkanVertexAttribs&& other) noexcept;
        GpuVertexAttribsInternal& operator=(GpuVertexAttribsInternal&& other) noexcept override;

        std::unique_ptr<GpuVertexAttribsInternal> Clone() const override;

        std::span<const VkVertexInputBindingDescription> InputBindings() const;
        std::span<const VkVertexInputAttributeDescription> InputAttribs() const;

    private:
        std::vector<VkVertexInputBindingDescription> input_bindings_;
        std::vector<VkVertexInputAttributeDescription> input_attribs_;
    };

    VULKAN_DEFINE_IMP(VertexAttribs)
} // namespace AIHoloImager
