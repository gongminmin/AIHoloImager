// Copyright (c) 2025-2026 Minmin Gong
//

#include "VulkanResourceViews.hpp"

#include <volk.h>

#include "VulkanBuffer.hpp"
#include "VulkanConversion.hpp"
#include "VulkanErrorHandling.hpp"
#include "VulkanSystem.hpp"
#include "VulkanTexture.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(ConstantBufferView)

    VulkanConstantBufferView::VulkanConstantBufferView(const GpuMemoryBlock& mem_block)
        : resource_(mem_block.Buffer()), mem_block_(&mem_block)
    {
        const VkBuffer vulkan_buff = VulkanImp(*mem_block.Buffer()).Buffer();

        buff_info_ = VkDescriptorBufferInfo{
            .buffer = vulkan_buff,
            .offset = mem_block.Offset(),
            .range = mem_block.Size(),
        };
        write_desc_set_ = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &buff_info_,
        };
    }

    VulkanConstantBufferView::~VulkanConstantBufferView() = default;

    VulkanConstantBufferView::VulkanConstantBufferView(VulkanConstantBufferView&& other) noexcept = default;
    VulkanConstantBufferView::VulkanConstantBufferView(GpuConstantBufferViewInternal&& other) noexcept
        : VulkanConstantBufferView(static_cast<VulkanConstantBufferView&&>(other))
    {
    }

    VulkanConstantBufferView& VulkanConstantBufferView::operator=(VulkanConstantBufferView&& other) noexcept = default;
    GpuConstantBufferViewInternal& VulkanConstantBufferView::operator=(GpuConstantBufferViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanConstantBufferView&&>(other));
    }

    void VulkanConstantBufferView::Reset()
    {
    }

    void VulkanConstantBufferView::Transition(GpuCommandList& cmd_list) const
    {
        this->Transition(VulkanImp(cmd_list));
    }

    void VulkanConstantBufferView::Transition(VulkanCommandList& cmd_list) const
    {
        cmd_list.RegisterAccessedObject(mem_block_->StalledWaitFences());
        cmd_list.RegisterAccessedObject(buff_view_.StalledWaitFences());

        VulkanImp(*resource_).Transition(cmd_list, GpuResourceState::Common);
    }

    VkWriteDescriptorSet VulkanConstantBufferView::WriteDescSet() const noexcept
    {
        return write_desc_set_;
    }


    VULKAN_IMP_IMP(ShaderResourceView)

    VulkanShaderResourceView::VulkanShaderResourceView(const GpuResource& resource) noexcept : resource_(&resource)
    {
    }
    VulkanShaderResourceView::~VulkanShaderResourceView() noexcept = default;
    VulkanShaderResourceView::VulkanShaderResourceView(VulkanShaderResourceView&& other) noexcept = default;
    VulkanShaderResourceView& VulkanShaderResourceView::operator=(VulkanShaderResourceView&& other) noexcept = default;

    void VulkanShaderResourceView::Transition(GpuCommandList& cmd_list) const
    {
        this->Transition(VulkanImp(cmd_list));
    }

    const GpuResource* VulkanShaderResourceView::Resource() const noexcept
    {
        return resource_;
    }

    VkWriteDescriptorSet VulkanShaderResourceView::WriteDescSet() const noexcept
    {
        return write_desc_set_;
    }


    VULKAN_IMP_IMP2(ShaderResourceView, BufferShaderResourceView)

    VulkanBufferShaderResourceView::VulkanBufferShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format)
        : VulkanShaderResourceView(buffer), buff_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = buff_view_.VulkanSys()->Device();
        const VkBuffer vulkan_buff = VulkanImp(buffer).Buffer();
        const uint32_t element_size = FormatSize(format);

        const uint32_t offset = first_element * element_size;
        const uint32_t range = num_elements * element_size;

        const GpuResourceFlag flags = buffer.Flags();
        if (EnumHasAny(flags, GpuResourceFlag::Structured))
        {
            this->FillWriteDescSetForStructuredBuffer(vulkan_buff, offset, range);
        }
        else
        {
            const VkBufferUsageFlags2CreateInfoKHR usage_create_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
                .usage = VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT_KHR,
            };
            const VkBufferViewCreateInfo view_create_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
                .pNext = &usage_create_info,
                .buffer = vulkan_buff,
                .format = ToVkFormat(format),
                .offset = offset,
                .range = range,
            };
            TIFVK(vkCreateBufferView(vulkan_device, &view_create_info, nullptr, &buff_view_.Object()));

            write_desc_set_ = VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
                .pTexelBufferView = &buff_view_.Object(),
            };
        }
    }

    VulkanBufferShaderResourceView::VulkanBufferShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size)
        : VulkanShaderResourceView(buffer), buff_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = buff_view_.VulkanSys()->Device();
        const VkBuffer vulkan_buff = VulkanImp(buffer).Buffer();

        const uint32_t offset = first_element * element_size;
        const uint32_t range = num_elements * element_size;

        const VkBufferUsageFlags2CreateInfoKHR usage_create_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
            .usage = VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR,
        };
        const VkBufferViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .pNext = &usage_create_info,
            .buffer = vulkan_buff,
            .format = VK_FORMAT_UNDEFINED,
            .offset = offset,
            .range = range,
        };
        TIFVK(vkCreateBufferView(vulkan_device, &view_create_info, nullptr, &buff_view_.Object()));

        this->FillWriteDescSetForStructuredBuffer(vulkan_buff, offset, range);
    }

    VulkanBufferShaderResourceView::~VulkanBufferShaderResourceView() = default;

    VulkanBufferShaderResourceView::VulkanBufferShaderResourceView(VulkanBufferShaderResourceView&& other) noexcept = default;
    VulkanBufferShaderResourceView::VulkanBufferShaderResourceView(GpuShaderResourceViewInternal&& other) noexcept
        : VulkanBufferShaderResourceView(static_cast<VulkanBufferShaderResourceView&&>(other))
    {
    }
    VulkanBufferShaderResourceView::VulkanBufferShaderResourceView(VulkanShaderResourceView&& other) noexcept
        : VulkanBufferShaderResourceView(static_cast<VulkanBufferShaderResourceView&&>(other))
    {
    }

    VulkanBufferShaderResourceView& VulkanBufferShaderResourceView::operator=(VulkanBufferShaderResourceView&& other) noexcept = default;
    GpuShaderResourceViewInternal& VulkanBufferShaderResourceView::operator=(GpuShaderResourceViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanBufferShaderResourceView&&>(other));
    }
    VulkanShaderResourceView& VulkanBufferShaderResourceView::operator=(VulkanShaderResourceView&& other) noexcept
    {
        return this->operator=(static_cast<VulkanBufferShaderResourceView&&>(other));
    }

    void VulkanBufferShaderResourceView::Reset()
    {
        buff_view_.Reset();
    }

    void VulkanBufferShaderResourceView::Transition(VulkanCommandList& cmd_list) const
    {
        cmd_list.RegisterAccessedObject(buff_view_.StalledWaitFences());
        VulkanImp(*resource_).Transition(cmd_list, 0, GpuResourceState::Common);
    }

    void VulkanBufferShaderResourceView::FillWriteDescSetForStructuredBuffer(VkBuffer buff, uint32_t offset, uint32_t range)
    {
        buff_info_ = VkDescriptorBufferInfo{
            .buffer = buff,
            .offset = offset,
            .range = range,
        };
        write_desc_set_ = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buff_info_,
        };
    }


    VULKAN_IMP_IMP2(ShaderResourceView, TextureShaderResourceView)

    VulkanTextureShaderResourceView::VulkanTextureShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format)
        : VulkanShaderResourceView(texture), sub_resource_(sub_resource), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture).Image();
        if ((format == GpuFormat::Unknown) || IsDepthStencilFormat(texture.Format()))
        {
            format = texture.Format();
        }

        const VkImageViewUsageCreateInfo usage_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        };
        VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usage_create_info,
            .image = vulkan_image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = ToVkFormat(format),
            .subresourceRange{
                .aspectMask = ToVulkanAspectMask(format),
            },
        };
        if (sub_resource == ~0U)
        {
            view_create_info.subresourceRange.baseMipLevel = 0;
            view_create_info.subresourceRange.levelCount = texture.MipLevels();
            view_create_info.subresourceRange.baseArrayLayer = 0;
            view_create_info.subresourceRange.layerCount = 1;
        }
        else
        {
            uint32_t array_slice;
            uint32_t plane_slice;
            DecomposeSubResource(
                sub_resource, texture.MipLevels(), 1, view_create_info.subresourceRange.baseMipLevel, array_slice, plane_slice);
            view_create_info.subresourceRange.levelCount = 1;
            view_create_info.subresourceRange.baseArrayLayer = 0;
            view_create_info.subresourceRange.layerCount = 1;
        }
        TIFVK(vkCreateImageView(vulkan_device, &view_create_info, nullptr, &image_view_.Object()));

        this->FillWriteDescSetForImage();
    }

    VulkanTextureShaderResourceView::VulkanTextureShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format)
        : VulkanShaderResourceView(texture_array), sub_resource_(sub_resource), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture_array).Image();
        if ((format == GpuFormat::Unknown) || IsDepthStencilFormat(texture_array.Format()))
        {
            format = texture_array.Format();
        }

        const VkImageViewUsageCreateInfo usage_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        };
        VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usage_create_info,
            .image = vulkan_image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .format = ToVkFormat(format),
            .subresourceRange{
                .aspectMask = ToVulkanAspectMask(format),
            },
        };
        if (sub_resource == ~0U)
        {
            view_create_info.subresourceRange.baseMipLevel = 0;
            view_create_info.subresourceRange.levelCount = texture_array.MipLevels();
            view_create_info.subresourceRange.baseArrayLayer = 0;
            view_create_info.subresourceRange.layerCount = texture_array.ArraySize();
        }
        else
        {
            uint32_t plane_slice;
            DecomposeSubResource(sub_resource, texture_array.MipLevels(), texture_array.ArraySize(),
                view_create_info.subresourceRange.baseMipLevel, view_create_info.subresourceRange.baseArrayLayer, plane_slice);
            view_create_info.subresourceRange.levelCount = 1;
            view_create_info.subresourceRange.layerCount = 1;
        }
        TIFVK(vkCreateImageView(vulkan_device, &view_create_info, nullptr, &image_view_.Object()));

        this->FillWriteDescSetForImage();
    }

    VulkanTextureShaderResourceView::VulkanTextureShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format)
        : VulkanShaderResourceView(texture), sub_resource_(sub_resource), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture).Image();
        if ((format == GpuFormat::Unknown) || IsDepthStencilFormat(texture.Format()))
        {
            format = texture.Format();
        }

        const VkImageViewUsageCreateInfo usage_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        };
        VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usage_create_info,
            .image = vulkan_image,
            .viewType = VK_IMAGE_VIEW_TYPE_3D,
            .format = ToVkFormat(format),
            .subresourceRange{
                .aspectMask = ToVulkanAspectMask(format),
            },
        };
        if (sub_resource == ~0U)
        {
            view_create_info.subresourceRange.baseMipLevel = 0;
            view_create_info.subresourceRange.levelCount = texture.MipLevels();
            view_create_info.subresourceRange.baseArrayLayer = 0;
            view_create_info.subresourceRange.layerCount = 1;
        }
        else
        {
            uint32_t array_slice;
            uint32_t plane_slice;
            DecomposeSubResource(
                sub_resource, texture.MipLevels(), 1, view_create_info.subresourceRange.baseMipLevel, array_slice, plane_slice);
            view_create_info.subresourceRange.levelCount = 1;
            view_create_info.subresourceRange.baseArrayLayer = 0;
            view_create_info.subresourceRange.layerCount = 1;
        }
        TIFVK(vkCreateImageView(vulkan_device, &view_create_info, nullptr, &image_view_.Object()));

        this->FillWriteDescSetForImage();
    }

    VulkanTextureShaderResourceView::~VulkanTextureShaderResourceView() = default;

    VulkanTextureShaderResourceView::VulkanTextureShaderResourceView(VulkanTextureShaderResourceView&& other) noexcept = default;
    VulkanTextureShaderResourceView::VulkanTextureShaderResourceView(GpuShaderResourceViewInternal&& other) noexcept
        : VulkanTextureShaderResourceView(static_cast<VulkanTextureShaderResourceView&&>(other))
    {
    }
    VulkanTextureShaderResourceView::VulkanTextureShaderResourceView(VulkanShaderResourceView&& other) noexcept
        : VulkanTextureShaderResourceView(static_cast<VulkanTextureShaderResourceView&&>(other))
    {
    }

    VulkanTextureShaderResourceView& VulkanTextureShaderResourceView::operator=(VulkanTextureShaderResourceView&& other) noexcept = default;
    GpuShaderResourceViewInternal& VulkanTextureShaderResourceView::operator=(GpuShaderResourceViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanTextureShaderResourceView&&>(other));
    }
    VulkanShaderResourceView& VulkanTextureShaderResourceView::operator=(VulkanShaderResourceView&& other) noexcept
    {
        return this->operator=(static_cast<VulkanTextureShaderResourceView&&>(other));
    }

    void VulkanTextureShaderResourceView::Reset()
    {
        image_view_.Reset();
    }

    void VulkanTextureShaderResourceView::Transition(VulkanCommandList& cmd_list) const
    {
        cmd_list.RegisterAccessedObject(image_view_.StalledWaitFences());
        VulkanImp(*resource_).Transition(cmd_list, sub_resource_, GpuResourceState::Common);
    }

    void VulkanTextureShaderResourceView::FillWriteDescSetForImage()
    {
        image_info_ = VkDescriptorImageInfo{
            .imageView = image_view_.Object(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        write_desc_set_ = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .pImageInfo = &image_info_,
        };
    }


    VULKAN_IMP_IMP(RenderTargetView)

    VulkanRenderTargetView::VulkanRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format)
        : resource_(&texture), sub_resource_(0), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture).Image();

        subres_range_ = VkImageSubresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        const VkImageViewUsageCreateInfo usage_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        };
        const VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usage_create_info,
            .image = vulkan_image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = ToVkFormat(format == GpuFormat::Unknown ? texture.Format() : format),
            .subresourceRange = subres_range_,
        };
        TIFVK(vkCreateImageView(vulkan_device, &view_create_info, nullptr, &image_view_.Object()));
    }

    VulkanRenderTargetView::~VulkanRenderTargetView()
    {
        this->Reset();
    }

    VulkanRenderTargetView::VulkanRenderTargetView(VulkanRenderTargetView&& other) noexcept = default;
    VulkanRenderTargetView::VulkanRenderTargetView(GpuRenderTargetViewInternal&& other) noexcept
        : VulkanRenderTargetView(static_cast<VulkanRenderTargetView&&>(other))
    {
    }
    VulkanRenderTargetView& VulkanRenderTargetView::operator=(VulkanRenderTargetView&& other) noexcept = default;
    GpuRenderTargetViewInternal& VulkanRenderTargetView::operator=(GpuRenderTargetViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanRenderTargetView&&>(other));
    }

    VulkanRenderTargetView::operator bool() const noexcept
    {
        return (image_view_.Object() != VK_NULL_HANDLE);
    }

    void VulkanRenderTargetView::Reset()
    {
        image_view_.Reset();
    }

    void VulkanRenderTargetView::Transition(GpuCommandList& cmd_list) const
    {
        this->Transition(VulkanImp(cmd_list));
    }

    void VulkanRenderTargetView::Transition(VulkanCommandList& cmd_list) const
    {
        cmd_list.RegisterAccessedObject(image_view_.StalledWaitFences());
        VulkanImp(*resource_).Transition(cmd_list, sub_resource_, GpuResourceState::ColorWrite);
    }

    void VulkanRenderTargetView::TransitionBack(VulkanCommandList& cmd_list) const
    {
        VulkanImp(*resource_).Transition(cmd_list, sub_resource_, GpuResourceState::Common);
    }

    GpuResource* VulkanRenderTargetView::Resource() const noexcept
    {
        return resource_;
    }

    VkImageView VulkanRenderTargetView::ImageView() const noexcept
    {
        return image_view_.Object();
    }

    VkImage VulkanRenderTargetView::Image() const noexcept
    {
        return static_cast<VulkanTexture&>(VulkanImp(*resource_)).Image();
    }

    const VkImageSubresourceRange& VulkanRenderTargetView::SubresourceRange() const
    {
        return subres_range_;
    }


    VULKAN_IMP_IMP(DepthStencilView)

    VulkanDepthStencilView::VulkanDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format)
        : resource_(&texture), sub_resource_(0), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture).Image();

        subres_range_ = VkImageSubresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        const VkImageViewUsageCreateInfo usage_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        };
        VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usage_create_info,
            .image = vulkan_image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = ToVkFormat(format == GpuFormat::Unknown ? texture.Format() : format),
            .subresourceRange = subres_range_,
        };
        if (IsStencilFormat(format))
        {
            view_create_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }

        TIFVK(vkCreateImageView(vulkan_device, &view_create_info, nullptr, &image_view_.Object()));
    }

    VulkanDepthStencilView::~VulkanDepthStencilView()
    {
        this->Reset();
    }

    VulkanDepthStencilView::VulkanDepthStencilView(VulkanDepthStencilView&& other) noexcept = default;
    VulkanDepthStencilView::VulkanDepthStencilView(GpuDepthStencilViewInternal&& other) noexcept
        : VulkanDepthStencilView(static_cast<VulkanDepthStencilView&&>(other))
    {
    }
    VulkanDepthStencilView& VulkanDepthStencilView::operator=(VulkanDepthStencilView&& other) noexcept = default;
    GpuDepthStencilViewInternal& VulkanDepthStencilView::operator=(GpuDepthStencilViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanDepthStencilView&&>(other));
    }

    VulkanDepthStencilView::operator bool() const noexcept
    {
        return (image_view_.Object() != VK_NULL_HANDLE);
    }

    void VulkanDepthStencilView::Reset()
    {
        image_view_.Reset();
    }

    void VulkanDepthStencilView::Transition(GpuCommandList& cmd_list) const
    {
        this->Transition(VulkanImp(cmd_list));
    }

    void VulkanDepthStencilView::Transition(VulkanCommandList& cmd_list) const
    {
        cmd_list.RegisterAccessedObject(image_view_.StalledWaitFences());
        VulkanImp(*resource_).Transition(cmd_list, sub_resource_, GpuResourceState::DepthWrite);
    }

    void VulkanDepthStencilView::TransitionBack(VulkanCommandList& cmd_list) const
    {
        VulkanImp(*resource_).Transition(cmd_list, sub_resource_, GpuResourceState::Common);
    }

    GpuResource* VulkanDepthStencilView::Resource() const noexcept
    {
        return resource_;
    }

    VkImageView VulkanDepthStencilView::ImageView() const noexcept
    {
        return image_view_.Object();
    }

    VkImage VulkanDepthStencilView::Image() const noexcept
    {
        return static_cast<VulkanTexture&>(VulkanImp(*resource_)).Image();
    }

    const VkImageSubresourceRange& VulkanDepthStencilView::SubresourceRange() const
    {
        return subres_range_;
    }


    VULKAN_IMP_IMP(UnorderedAccessView)

    VulkanUnorderedAccessView::VulkanUnorderedAccessView(GpuResource& resource) noexcept : resource_(&resource)
    {
    }
    VulkanUnorderedAccessView::~VulkanUnorderedAccessView() noexcept = default;
    VulkanUnorderedAccessView::VulkanUnorderedAccessView(VulkanUnorderedAccessView&& other) noexcept = default;
    VulkanUnorderedAccessView& VulkanUnorderedAccessView::operator=(VulkanUnorderedAccessView&& other) noexcept = default;

    void VulkanUnorderedAccessView::Transition(GpuCommandList& cmd_list) const
    {
        this->Transition(VulkanImp(cmd_list));
    }

    GpuResource* VulkanUnorderedAccessView::Resource() noexcept
    {
        return resource_;
    }

    VkWriteDescriptorSet VulkanUnorderedAccessView::WriteDescSet() const noexcept
    {
        return write_desc_set_;
    }


    VULKAN_IMP_IMP2(UnorderedAccessView, BufferUnorderedAccessView)

    VulkanBufferUnorderedAccessView::VulkanBufferUnorderedAccessView(
        GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format)
        : VulkanUnorderedAccessView(buffer), buff_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = buff_view_.VulkanSys()->Device();
        const VkBuffer vulkan_buff = VulkanImp(buffer).Buffer();
        const uint32_t element_size = FormatSize(format);

        buff_range_ = {first_element * element_size, num_elements * element_size};

        const GpuResourceFlag flags = buffer.Flags();
        if (EnumHasAny(flags, GpuResourceFlag::Structured))
        {
            this->FillWriteDescSetForStructuredBuffer(vulkan_buff);
        }
        else
        {
            const VkBufferUsageFlags2CreateInfoKHR usage_create_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
                .usage = VK_BUFFER_USAGE_2_STORAGE_TEXEL_BUFFER_BIT_KHR,
            };
            const VkBufferViewCreateInfo view_create_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
                .pNext = &usage_create_info,
                .buffer = vulkan_buff,
                .format = ToVkFormat(format),
                .offset = std::get<0>(buff_range_),
                .range = std::get<1>(buff_range_),
            };
            TIFVK(vkCreateBufferView(vulkan_device, &view_create_info, nullptr, &buff_view_.Object()));

            write_desc_set_ = VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
                .pTexelBufferView = &buff_view_.Object(),
            };
        }
    }

    VulkanBufferUnorderedAccessView::VulkanBufferUnorderedAccessView(
        GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size)
        : VulkanUnorderedAccessView(buffer), buff_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkBuffer vulkan_buff = VulkanImp(buffer).Buffer();

        buff_range_ = {first_element * element_size, num_elements * element_size};

        this->FillWriteDescSetForStructuredBuffer(vulkan_buff);
    }

    VulkanBufferUnorderedAccessView::~VulkanBufferUnorderedAccessView()
    {
        this->Reset();
    }

    VulkanBufferUnorderedAccessView::VulkanBufferUnorderedAccessView(VulkanBufferUnorderedAccessView&& other) noexcept = default;
    VulkanBufferUnorderedAccessView::VulkanBufferUnorderedAccessView(GpuUnorderedAccessViewInternal&& other) noexcept
        : VulkanBufferUnorderedAccessView(static_cast<VulkanBufferUnorderedAccessView&&>(other))
    {
    }
    VulkanBufferUnorderedAccessView::VulkanBufferUnorderedAccessView(VulkanUnorderedAccessView&& other) noexcept
        : VulkanBufferUnorderedAccessView(static_cast<VulkanBufferUnorderedAccessView&&>(other))
    {
    }
    VulkanBufferUnorderedAccessView& VulkanBufferUnorderedAccessView::operator=(VulkanBufferUnorderedAccessView&& other) noexcept = default;
    GpuUnorderedAccessViewInternal& VulkanBufferUnorderedAccessView::operator=(GpuUnorderedAccessViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanBufferUnorderedAccessView&&>(other));
    }
    VulkanUnorderedAccessView& VulkanBufferUnorderedAccessView::operator=(VulkanUnorderedAccessView&& other) noexcept
    {
        return this->operator=(static_cast<VulkanBufferUnorderedAccessView&&>(other));
    }

    void VulkanBufferUnorderedAccessView::Reset()
    {
        buff_view_.Reset();
    }

    void VulkanBufferUnorderedAccessView::Transition(VulkanCommandList& cmd_list) const
    {
        cmd_list.RegisterAccessedObject(buff_view_.StalledWaitFences());
        VulkanImp(*resource_).Transition(cmd_list, 0, GpuResourceState::UnorderedAccess);
    }

    VkBuffer VulkanBufferUnorderedAccessView::Buffer() const noexcept
    {
        return static_cast<VulkanBuffer&>(VulkanImp(*resource_)).Buffer();
    }
    VkBufferView VulkanBufferUnorderedAccessView::BufferView() const noexcept
    {
        return buff_view_.Object();
    }
    const std::tuple<uint32_t, uint32_t>& VulkanBufferUnorderedAccessView::BufferRange() const
    {
        return buff_range_;
    }

    void VulkanBufferUnorderedAccessView::FillWriteDescSetForStructuredBuffer(VkBuffer buff)
    {
        buffer_info_ = VkDescriptorBufferInfo{
            .buffer = buff,
            .offset = std::get<0>(buff_range_),
            .range = std::get<1>(buff_range_),
        };
        write_desc_set_ = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_info_,
        };
    }


    VULKAN_IMP_IMP2(UnorderedAccessView, TextureUnorderedAccessView)

    VulkanTextureUnorderedAccessView::VulkanTextureUnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format)
        : VulkanUnorderedAccessView(texture), sub_resource_(sub_resource), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture).Image();
        if ((format == GpuFormat::Unknown) || IsDepthStencilFormat(texture.Format()))
        {
            format = texture.Format();
        }

        subres_range_ = VkImageSubresourceRange{
            .aspectMask = ToVulkanAspectMask(format),
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, texture.MipLevels(), 1, subres_range_.baseMipLevel, array_slice, plane_slice);

        this->CreateImageView(vulkan_device, vulkan_image, VK_IMAGE_VIEW_TYPE_2D, format);
        this->FillWriteDescSetForImage();
    }

    VulkanTextureUnorderedAccessView::VulkanTextureUnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format)
        : VulkanUnorderedAccessView(texture_array), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture_array).Image();
        if ((format == GpuFormat::Unknown) || IsDepthStencilFormat(texture_array.Format()))
        {
            format = texture_array.Format();
        }

        subres_range_ = VkImageSubresourceRange{
            .aspectMask = ToVulkanAspectMask(format),
            .levelCount = 1,
            .layerCount = 1,
        };

        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, texture_array.MipLevels(), texture_array.ArraySize(), subres_range_.baseMipLevel,
            subres_range_.baseArrayLayer, plane_slice);

        this->CreateImageView(vulkan_device, vulkan_image, VK_IMAGE_VIEW_TYPE_2D_ARRAY, format);
        this->FillWriteDescSetForImage();
    }

    VulkanTextureUnorderedAccessView::VulkanTextureUnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format)
        : VulkanUnorderedAccessView(texture), sub_resource_(sub_resource), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture).Image();
        if ((format == GpuFormat::Unknown) || IsDepthStencilFormat(texture.Format()))
        {
            format = texture.Format();
        }

        subres_range_ = VkImageSubresourceRange{
            .aspectMask = ToVulkanAspectMask(format),
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, texture.MipLevels(), 1, subres_range_.baseMipLevel, array_slice, plane_slice);

        this->CreateImageView(vulkan_device, vulkan_image, VK_IMAGE_VIEW_TYPE_3D, format);
        this->FillWriteDescSetForImage();
    }

    VulkanTextureUnorderedAccessView::~VulkanTextureUnorderedAccessView()
    {
        this->Reset();
    }

    VulkanTextureUnorderedAccessView::VulkanTextureUnorderedAccessView(VulkanTextureUnorderedAccessView&& other) noexcept = default;
    VulkanTextureUnorderedAccessView::VulkanTextureUnorderedAccessView(GpuUnorderedAccessViewInternal&& other) noexcept
        : VulkanTextureUnorderedAccessView(static_cast<VulkanTextureUnorderedAccessView&&>(other))
    {
    }
    VulkanTextureUnorderedAccessView::VulkanTextureUnorderedAccessView(VulkanUnorderedAccessView&& other) noexcept
        : VulkanTextureUnorderedAccessView(static_cast<VulkanTextureUnorderedAccessView&&>(other))
    {
    }
    VulkanTextureUnorderedAccessView& VulkanTextureUnorderedAccessView::operator=(
        VulkanTextureUnorderedAccessView&& other) noexcept = default;
    GpuUnorderedAccessViewInternal& VulkanTextureUnorderedAccessView::operator=(GpuUnorderedAccessViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanTextureUnorderedAccessView&&>(other));
    }
    VulkanUnorderedAccessView& VulkanTextureUnorderedAccessView::operator=(VulkanUnorderedAccessView&& other) noexcept
    {
        return this->operator=(static_cast<VulkanTextureUnorderedAccessView&&>(other));
    }

    void VulkanTextureUnorderedAccessView::Reset()
    {
        image_view_.Reset();
    }

    void VulkanTextureUnorderedAccessView::Transition(VulkanCommandList& cmd_list) const
    {
        cmd_list.RegisterAccessedObject(image_view_.StalledWaitFences());
        VulkanImp(*resource_).Transition(cmd_list, sub_resource_, GpuResourceState::UnorderedAccess);
    }

    VkImage VulkanTextureUnorderedAccessView::Image() const noexcept
    {
        return static_cast<VulkanTexture&>(VulkanImp(*resource_)).Image();
    }
    VkImageView VulkanTextureUnorderedAccessView::ImageView() const noexcept
    {
        return image_view_.Object();
    }
    const VkImageSubresourceRange& VulkanTextureUnorderedAccessView::SubresourceRange() const
    {
        return subres_range_;
    }

    void VulkanTextureUnorderedAccessView::CreateImageView(VkDevice device, VkImage image, VkImageViewType type, GpuFormat format)
    {
        const VkImageViewUsageCreateInfo usage_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT,
        };
        const VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usage_create_info,
            .image = image,
            .viewType = type,
            .format = ToVkFormat(format),
            .subresourceRange = subres_range_,
        };
        TIFVK(vkCreateImageView(device, &view_create_info, nullptr, &image_view_.Object()));
    }

    void VulkanTextureUnorderedAccessView::FillWriteDescSetForImage()
    {
        image_info_ = VkDescriptorImageInfo{
            .imageView = image_view_.Object(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        write_desc_set_ = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &image_info_,
        };
    }


    VkWriteDescriptorSet NullSampledImageShaderResourceViewWriteDescSet()
    {
        static VkDescriptorImageInfo null_image_info{
            .imageView = VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        static VkWriteDescriptorSet null_write_desc_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .pImageInfo = &null_image_info,
        };
        return null_write_desc_set;
    }
    VkWriteDescriptorSet NullUniformTexelBufferShaderResourceViewWriteDescSet()
    {
        static VkBufferView null_buff_view = VK_NULL_HANDLE;
        static VkWriteDescriptorSet null_write_desc_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
            .pTexelBufferView = &null_buff_view,
        };
        return null_write_desc_set;
    }

    VkWriteDescriptorSet NullStorageImageUnorderedAccessViewWriteDescSet()
    {
        static VkDescriptorImageInfo null_image_info{
            .imageView = VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        static VkWriteDescriptorSet null_write_desc_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &null_image_info,
        };
        return null_write_desc_set;
    }
    VkWriteDescriptorSet NullStorageTexelBufferUnorderedAccessViewWriteDescSet()
    {
        static VkBufferView null_buff_view = VK_NULL_HANDLE;
        static VkWriteDescriptorSet null_write_desc_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
            .pTexelBufferView = &null_buff_view,
        };
        return null_write_desc_set;
    }
    VkWriteDescriptorSet NullStorageBufferUnorderedAccessViewWriteDescSet()
    {
        static VkDescriptorBufferInfo null_buff_info{
            .buffer = VK_NULL_HANDLE,
        };
        static VkWriteDescriptorSet null_write_desc_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &null_buff_info,
        };
        return null_write_desc_set;
    }

    VkWriteDescriptorSet NullUniformBufferConstantBufferViewWriteDescSet()
    {
        static VkDescriptorBufferInfo null_buff_info{
            .buffer = VK_NULL_HANDLE,
        };
        static VkWriteDescriptorSet null_write_desc_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &null_buff_info,
        };
        return null_write_desc_set;
    }
} // namespace AIHoloImager
