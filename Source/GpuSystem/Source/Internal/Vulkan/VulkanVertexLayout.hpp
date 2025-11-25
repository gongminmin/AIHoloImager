// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <span>
#include <vector>

#include <volk.h>

#include "Gpu/GpuVertexLayout.hpp"

#include "../GpuVertexLayoutInternal.hpp"
#include "VulkanImpDefine.hpp"

namespace AIHoloImager
{
    class VulkanVertexLayout : public GpuVertexLayoutInternal
    {
    public:
        explicit VulkanVertexLayout(std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides = {});
        ~VulkanVertexLayout() override;

        VulkanVertexLayout(const VulkanVertexLayout& other);
        explicit VulkanVertexLayout(const GpuVertexLayoutInternal& other);

        VulkanVertexLayout& operator=(const VulkanVertexLayout& other);
        GpuVertexLayoutInternal& operator=(const GpuVertexLayoutInternal& other) override;

        VulkanVertexLayout(VulkanVertexLayout&& other) noexcept;
        explicit VulkanVertexLayout(GpuVertexLayoutInternal&& other) noexcept;

        VulkanVertexLayout& operator=(VulkanVertexLayout&& other) noexcept;
        GpuVertexLayoutInternal& operator=(GpuVertexLayoutInternal&& other) noexcept override;

        std::unique_ptr<GpuVertexLayoutInternal> Clone() const override;

        std::span<const VkVertexInputBindingDescription> InputBindings() const;
        std::span<const VkVertexInputAttributeDescription> InputAttribs() const;

    private:
        std::vector<VkVertexInputBindingDescription> input_bindings_;
        std::vector<VkVertexInputAttributeDescription> input_attribs_;
    };

    VULKAN_DEFINE_IMP(VertexLayout)
} // namespace AIHoloImager
