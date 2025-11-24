// Copyright (c) 2025 Minmin Gong
//

#include "VulkanTexture.hpp"

#include <cassert>

#include "VulkanCommandList.hpp"
#include "VulkanConversion.hpp"
#include "VulkanErrorhandling.hpp"
#include "VulkanSystem.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(Texture)

    VulkanTexture::VulkanTexture(GpuSystem& gpu_system, GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
        uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::string_view name)
        : VulkanResource(gpu_system), format_(format), flags_(flags), texture_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        assert(mip_levels != 0);

        const auto& vulkan_system = *texture_.VulkanSys();
        const VkDevice vulkan_device = vulkan_system.Device();

        curr_layouts_.resize(array_size * mip_levels * NumPlanes(format), VK_IMAGE_LAYOUT_UNDEFINED);

        image_create_info_ = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = ToVulkanImageType(type),
            .format = ToVkFormat(format),
            .extent{
                .width = width,
                .height = height,
                .depth = depth,
            },
            .mipLevels = mip_levels,
            .arrayLayers = array_size,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = ToVulkanImageUsageFlags(flags_),
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        VkExternalMemoryImageCreateInfo external_mem_image_create_info;
        if (EnumHasAny(flags, GpuResourceFlag::Shareable))
        {
            external_mem_image_create_info = {
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };

            image_create_info_.pNext = &external_mem_image_create_info;
        }

        TIFVK(vkCreateImage(vulkan_device, &image_create_info_, nullptr, &texture_.Object()));

        VkMemoryRequirements requirements;
        vkGetImageMemoryRequirements(vulkan_device, texture_.Object(), &requirements);
        this->CreateMemory(type, requirements, GpuHeap::Default, flags);
        TIFVK(vkBindImageMemory(vulkan_device, texture_.Object(), this->Memory(), 0));

        this->Name(std::move(name));
    }

    VulkanTexture::VulkanTexture(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::string_view name)
        : VulkanResource(gpu_system), texture_(VulkanImp(gpu_system), reinterpret_cast<VkImage>(native_resource))
    {
        if (texture_.Object() != VK_NULL_HANDLE)
        {
            // desc_ = resource_->GetDesc();
            this->Name(std::move(name));

            /*switch (desc_.Dimension)
            {
            case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
                if (desc_.DepthOrArraySize > 1)
                {
                    type_ = GpuResourceType::Texture2DArray;
                }
                else
                {
                    type_ = GpuResourceType::Texture2D;
                }
                break;
            case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
                type_ = GpuResourceType::Texture3D;
                break;
            default:
                Unreachable("Invalid resource dimension");
            }*/

            curr_layouts_.assign(this->MipLevels() * this->Planes(), ToVulkanImageLayout(curr_state));
            format_ = this->Format();
            flags_ = this->Flags();
        }
    }

    VulkanTexture::~VulkanTexture() = default;

    VulkanTexture::VulkanTexture(VulkanTexture&& other) noexcept = default;
    VulkanTexture::VulkanTexture(GpuResourceInternal&& other) noexcept : VulkanTexture(static_cast<VulkanTexture&&>(other))
    {
    }
    VulkanTexture::VulkanTexture(GpuTextureInternal&& other) noexcept : VulkanTexture(static_cast<VulkanTexture&&>(other))
    {
    }
    VulkanTexture& VulkanTexture::operator=(VulkanTexture&& other) noexcept = default;
    GpuResourceInternal& VulkanTexture::operator=(GpuResourceInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanTexture&&>(other));
    }
    GpuTextureInternal& VulkanTexture::operator=(GpuTextureInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanTexture&&>(other));
    }

    void VulkanTexture::Name(std::string_view name)
    {
        this->VulkanResource::Name(texture_.Object(), std::move(name));
    }

    VkImage VulkanTexture::Image() const noexcept
    {
        return texture_.Object();
    }

    void* VulkanTexture::NativeResource() const noexcept
    {
        return this->Image();
    }

    void* VulkanTexture::NativeTexture() const noexcept
    {
        return this->NativeResource();
    }

    void* VulkanTexture::SharedHandle() const noexcept
    {
        return this->VulkanResource::SharedHandle();
    }

    GpuResourceType VulkanTexture::Type() const noexcept
    {
        return this->VulkanResource::Type();
    }

    uint32_t VulkanTexture::AllocationSize() const noexcept
    {
        const VkDevice vulkan_device = texture_.VulkanSys()->Device();

        VkMemoryRequirements requirements;
        vkGetImageMemoryRequirements(vulkan_device, texture_.Object(), &requirements);
        return static_cast<uint32_t>(requirements.size);
    }

    GpuResourceFlag VulkanTexture::Flags() const noexcept
    {
        return this->VulkanResource::Flags();
    }

    uint32_t VulkanTexture::Width(uint32_t mip) const noexcept
    {
        return std::max(image_create_info_.extent.width >> mip, 1U);
    }

    uint32_t VulkanTexture::Height(uint32_t mip) const noexcept
    {
        return std::max(image_create_info_.extent.height >> mip, 1U);
    }

    uint32_t VulkanTexture::Depth(uint32_t mip) const noexcept
    {
        return std::max(image_create_info_.extent.depth >> mip, 1U);
    }

    uint32_t VulkanTexture::ArraySize() const noexcept
    {
        return image_create_info_.arrayLayers;
    }

    uint32_t VulkanTexture::MipLevels() const noexcept
    {
        return image_create_info_.mipLevels;
    }

    uint32_t VulkanTexture::Planes() const noexcept
    {
        return NumPlanes(format_);
    }

    GpuFormat VulkanTexture::Format() const noexcept
    {
        return format_;
    }

    void VulkanTexture::Reset()
    {
        this->VulkanResource::Reset();

        texture_.Reset();
        image_create_info_ = {};
        curr_layouts_.clear();
    }

    void VulkanTexture::Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const
    {
        this->Transition(VulkanImp(cmd_list), sub_resource, target_state);
    }

    void VulkanTexture::Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const
    {
        this->Transition(VulkanImp(cmd_list), target_state);
    }

    void VulkanTexture::Transition(VulkanCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const
    {
        if (sub_resource == ~0U)
        {
            this->Transition(cmd_list, target_state);
        }
        else
        {
            auto vulkan_target_layout = ToVulkanImageLayout(target_state);
            if (curr_layouts_[sub_resource] != vulkan_target_layout)
            {
                uint32_t mip_level, array_index, plane;
                DecomposeSubResource(sub_resource, this->MipLevels(), this->Planes(), mip_level, array_index, plane);

                const VkImageMemoryBarrier barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    .oldLayout = curr_layouts_[sub_resource],
                    .newLayout = vulkan_target_layout,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = texture_.Object(),
                    .subresourceRange{
                        .aspectMask = ToVulkanAspectMask(format_),
                        .baseMipLevel = mip_level,
                        .levelCount = 1,
                        .baseArrayLayer = array_index,
                        .layerCount = 1,
                    },
                };
                cmd_list.Transition(std::span(&barrier, 1));

                curr_layouts_[sub_resource] = vulkan_target_layout;
            }
        }
    }

    void VulkanTexture::Transition(VulkanCommandList& cmd_list, GpuResourceState target_state) const
    {
        const VkImageLayout vulkan_target_layout = ToVulkanImageLayout(target_state);
        if ((curr_layouts_[0] == vulkan_target_layout) &&
            ((target_state == GpuResourceState::UnorderedAccess) || (target_state == GpuResourceState::RayTracingAS)))
        {
            VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = texture_.Object(),
                .subresourceRange{
                    .aspectMask = ToVulkanAspectMask(format_),
                    .baseMipLevel = 0,
                    .levelCount = this->MipLevels(),
                    .baseArrayLayer = 0,
                    .layerCount = this->ArraySize(),
                },
            };
            std::tie(barrier.srcAccessMask, barrier.dstAccessMask) = ToVulkanAccessFlags(barrier.oldLayout, barrier.newLayout);

            cmd_list.Transition(std::span(&barrier, 1));
        }
        else
        {
            bool same_state = true;
            for (size_t i = 1; i < curr_layouts_.size(); ++i)
            {
                if (curr_layouts_[i] != curr_layouts_[0])
                {
                    same_state = false;
                    break;
                }
            }

            if (same_state)
            {
                if (curr_layouts_[0] != vulkan_target_layout)
                {
                    VkImageMemoryBarrier barrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .oldLayout = curr_layouts_[0],
                        .newLayout = vulkan_target_layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = texture_.Object(),
                        .subresourceRange{
                            .aspectMask = ToVulkanAspectMask(format_),
                            .baseMipLevel = 0,
                            .levelCount = this->MipLevels(),
                            .baseArrayLayer = 0,
                            .layerCount = this->ArraySize(),
                        },
                    };
                    std::tie(barrier.srcAccessMask, barrier.dstAccessMask) = ToVulkanAccessFlags(barrier.oldLayout, barrier.newLayout);

                    cmd_list.Transition(std::span(&barrier, 1));
                }
            }
            else
            {
                std::vector<VkImageMemoryBarrier> barriers;
                for (size_t i = 0; i < curr_layouts_.size(); ++i)
                {
                    if (curr_layouts_[i] != vulkan_target_layout)
                    {
                        uint32_t mip_level, array_index, plane;
                        DecomposeSubResource(static_cast<uint32_t>(i), this->MipLevels(), this->Planes(), mip_level, array_index, plane);

                        auto& barrier = barriers.emplace_back();
                        barrier = {
                            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            .oldLayout = curr_layouts_[i],
                            .newLayout = vulkan_target_layout,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .image = texture_.Object(),
                            .subresourceRange{
                                .aspectMask = ToVulkanAspectMask(format_),
                                .baseMipLevel = mip_level,
                                .levelCount = 1,
                                .baseArrayLayer = array_index,
                                .layerCount = 1,
                            },
                        };
                        std::tie(barrier.srcAccessMask, barrier.dstAccessMask) = ToVulkanAccessFlags(barrier.oldLayout, barrier.newLayout);
                    }
                }
                cmd_list.Transition(std::span(barriers.begin(), barriers.end()));
            }
        }

        curr_layouts_.assign(this->MipLevels() * this->Planes(), vulkan_target_layout);
    }
} // namespace AIHoloImager
