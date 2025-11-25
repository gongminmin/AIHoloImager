// Copyright (c) 2025 Minmin Gong
//

#include "VulkanVertexLayout.hpp"

#include <map>

#include "Gpu/GpuFormat.hpp"

#include "VulkanConversion.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(VertexLayout)

    VulkanVertexLayout::VulkanVertexLayout(std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides)
        : input_attribs_(attribs.size())
    {
        std::map<uint32_t, uint32_t> slot_size;
        for (size_t i = 0; i < attribs.size(); ++i)
        {
            input_attribs_[i] = VkVertexInputAttributeDescription{
                .location = static_cast<uint32_t>(i),
                .binding = attribs[i].slot,
                .format = ToVkFormat(attribs[i].format),
            };

            auto iter = slot_size.find(attribs[i].slot);
            if (iter == slot_size.end())
            {
                iter = slot_size.emplace(attribs[i].slot, 0).first;
            }
            if (attribs[i].offset == GpuVertexAttrib::AppendOffset)
            {
                input_attribs_[i].offset = iter->second;
            }
            else
            {
                input_attribs_[i].offset = attribs[i].offset;
            }
            iter->second = std::max(iter->second, input_attribs_[i].offset + FormatSize(attribs[i].format));
        }

        if (slot_strides.empty())
        {
            for (size_t i = 0; i < attribs.size(); ++i)
            {
                bool found = false;
                for (size_t bi = 0; bi < input_bindings_.size(); ++bi)
                {
                    if (input_bindings_[bi].binding == attribs[i].slot)
                    {
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    input_bindings_.emplace_back(VkVertexInputBindingDescription{
                        .binding = attribs[i].slot,
                        .stride = slot_size[attribs[i].slot],
                        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
                    });
                }
            }
        }
        else
        {
            for (size_t i = 0; i < slot_strides.size(); ++i)
            {
                input_bindings_.emplace_back(VkVertexInputBindingDescription{
                    .binding = static_cast<uint32_t>(i),
                    .stride = slot_strides[i],
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
                });
            }
        }
    }

    VulkanVertexLayout::~VulkanVertexLayout() = default;

    VulkanVertexLayout::VulkanVertexLayout(const VulkanVertexLayout& other) = default;
    VulkanVertexLayout::VulkanVertexLayout(const GpuVertexLayoutInternal& other)
        : VulkanVertexLayout(static_cast<const VulkanVertexLayout&>(other))
    {
    }

    VulkanVertexLayout& VulkanVertexLayout::operator=(const VulkanVertexLayout& other) = default;
    GpuVertexLayoutInternal& VulkanVertexLayout::operator=(const GpuVertexLayoutInternal& other)
    {
        return this->operator=(static_cast<const VulkanVertexLayout&>(other));
    }

    VulkanVertexLayout::VulkanVertexLayout(VulkanVertexLayout&& other) noexcept = default;
    VulkanVertexLayout::VulkanVertexLayout(GpuVertexLayoutInternal&& other) noexcept
        : VulkanVertexLayout(static_cast<VulkanVertexLayout&&>(other))
    {
    }

    VulkanVertexLayout& VulkanVertexLayout::operator=(VulkanVertexLayout&& other) noexcept = default;
    GpuVertexLayoutInternal& VulkanVertexLayout::operator=(GpuVertexLayoutInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanVertexLayout&&>(other));
    }

    std::unique_ptr<GpuVertexLayoutInternal> VulkanVertexLayout::Clone() const
    {
        return std::make_unique<VulkanVertexLayout>(*this);
    }

    std::span<const VkVertexInputBindingDescription> VulkanVertexLayout::InputBindings() const
    {
        return input_bindings_;
    }

    std::span<const VkVertexInputAttributeDescription> VulkanVertexLayout::InputAttribs() const
    {
        return input_attribs_;
    }
} // namespace AIHoloImager
