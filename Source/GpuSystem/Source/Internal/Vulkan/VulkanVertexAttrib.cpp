// Copyright (c) 2025 Minmin Gong
//

#include "VulkanVertexAttrib.hpp"

#include <map>

#include "Gpu/GpuFormat.hpp"

#include "VulkanConversion.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(VertexAttribs)

    VulkanVertexAttribs::VulkanVertexAttribs(std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides)
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

    VulkanVertexAttribs::~VulkanVertexAttribs() = default;

    VulkanVertexAttribs::VulkanVertexAttribs(const VulkanVertexAttribs& other) = default;
    VulkanVertexAttribs::VulkanVertexAttribs(const GpuVertexAttribsInternal& other)
        : VulkanVertexAttribs(static_cast<const VulkanVertexAttribs&>(other))
    {
    }

    VulkanVertexAttribs& VulkanVertexAttribs::operator=(const VulkanVertexAttribs& other) = default;
    GpuVertexAttribsInternal& VulkanVertexAttribs::operator=(const GpuVertexAttribsInternal& other)
    {
        return this->operator=(static_cast<const VulkanVertexAttribs&>(other));
    }

    VulkanVertexAttribs::VulkanVertexAttribs(VulkanVertexAttribs&& other) noexcept = default;
    VulkanVertexAttribs::VulkanVertexAttribs(GpuVertexAttribsInternal&& other) noexcept
        : VulkanVertexAttribs(static_cast<VulkanVertexAttribs&&>(other))
    {
    }

    VulkanVertexAttribs& VulkanVertexAttribs::operator=(VulkanVertexAttribs&& other) noexcept = default;
    GpuVertexAttribsInternal& VulkanVertexAttribs::operator=(GpuVertexAttribsInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanVertexAttribs&&>(other));
    }

    std::unique_ptr<GpuVertexAttribsInternal> VulkanVertexAttribs::Clone() const
    {
        return std::make_unique<VulkanVertexAttribs>(*this);
    }

    std::span<const VkVertexInputBindingDescription> VulkanVertexAttribs::InputBindings() const
    {
        return std::span(input_bindings_);
    }

    std::span<const VkVertexInputAttributeDescription> VulkanVertexAttribs::InputAttribs() const
    {
        return std::span(input_attribs_);
    }
} // namespace AIHoloImager
