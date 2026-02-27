// Copyright (c) 2025-2026 Minmin Gong
//

#include "VulkanResourceViews.hpp"

#include <volk.h>

#include "VulkanBuffer.hpp"
#include "VulkanConversion.hpp"
#include "VulkanErrorhandling.hpp"
#include "VulkanSystem.hpp"
#include "VulkanTexture.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(ConstantBufferView)

    VulkanConstantBufferView::VulkanConstantBufferView(const GpuBuffer& buffer, uint32_t offset, uint32_t size) : resource_(&buffer)
    {
        const VkBuffer vulkan_buff = VulkanImp(buffer).Buffer();

        buff_info_ = VkDescriptorBufferInfo{
            .buffer = vulkan_buff,
            .offset = offset,
            .range = size,
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
        VulkanImp(*resource_).Transition(cmd_list, GpuResourceState::Common);
    }

    VkWriteDescriptorSet VulkanConstantBufferView::WriteDescSet() const noexcept
    {
        return write_desc_set_;
    }


    VULKAN_IMP_IMP(ShaderResourceView)

    VulkanShaderResourceView::VulkanShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format)
        : resource_(&texture), sub_resource_(sub_resource), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture).Image();
        if ((format == GpuFormat::Unknown) || IsDepthStencilFormat(texture.Format()))
        {
            format = texture.Format();
        }

        VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
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

    VulkanShaderResourceView::VulkanShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format)
        : resource_(&texture_array), sub_resource_(sub_resource), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture_array).Image();
        if ((format == GpuFormat::Unknown) || IsDepthStencilFormat(texture_array.Format()))
        {
            format = texture_array.Format();
        }

        VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
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

    VulkanShaderResourceView::VulkanShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format)
        : resource_(&texture), sub_resource_(sub_resource), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = image_view_.VulkanSys()->Device();
        const VkImage vulkan_image = VulkanImp(texture).Image();
        if ((format == GpuFormat::Unknown) || IsDepthStencilFormat(texture.Format()))
        {
            format = texture.Format();
        }

        VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
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

    VulkanShaderResourceView::VulkanShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format)
        : resource_(&buffer), sub_resource_(0), buff_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
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
            const VkBufferViewCreateInfo view_create_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
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

    VulkanShaderResourceView::VulkanShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size)
        : resource_(&buffer), sub_resource_(0), buff_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = buff_view_.VulkanSys()->Device();
        const VkBuffer vulkan_buff = VulkanImp(buffer).Buffer();

        const uint32_t offset = first_element * element_size;
        const uint32_t range = num_elements * element_size;

        const VkBufferViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .buffer = vulkan_buff,
            .format = VK_FORMAT_UNDEFINED,
            .offset = offset,
            .range = range,
        };
        TIFVK(vkCreateBufferView(vulkan_device, &view_create_info, nullptr, &buff_view_.Object()));

        this->FillWriteDescSetForStructuredBuffer(vulkan_buff, offset, range);
    }

    VulkanShaderResourceView::~VulkanShaderResourceView() = default;

    VulkanShaderResourceView::VulkanShaderResourceView(VulkanShaderResourceView&& other) noexcept = default;
    VulkanShaderResourceView::VulkanShaderResourceView(GpuShaderResourceViewInternal&& other) noexcept
        : VulkanShaderResourceView(static_cast<VulkanShaderResourceView&&>(other))
    {
    }

    VulkanShaderResourceView& VulkanShaderResourceView::operator=(VulkanShaderResourceView&& other) noexcept = default;
    GpuShaderResourceViewInternal& VulkanShaderResourceView::operator=(GpuShaderResourceViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanShaderResourceView&&>(other));
    }

    void VulkanShaderResourceView::Reset()
    {
        buff_view_.Reset();
        image_view_.Reset();
    }

    void VulkanShaderResourceView::Transition(GpuCommandList& cmd_list) const
    {
        this->Transition(VulkanImp(cmd_list));
    }

    void VulkanShaderResourceView::Transition(VulkanCommandList& cmd_list) const
    {
        VulkanImp(*resource_).Transition(cmd_list, sub_resource_, GpuResourceState::Common);
    }

    VkWriteDescriptorSet VulkanShaderResourceView::WriteDescSet() const noexcept
    {
        return write_desc_set_;
    }

    const GpuResource* VulkanShaderResourceView::Resource() const noexcept
    {
        return resource_;
    }

    void VulkanShaderResourceView::FillWriteDescSetForImage()
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

    void VulkanShaderResourceView::FillWriteDescSetForStructuredBuffer(VkBuffer buff, uint32_t offset, uint32_t range)
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

        const VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
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

        VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
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

    VulkanUnorderedAccessView::VulkanUnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format)
        : resource_(&texture), sub_resource_(sub_resource), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE), buff_range_({0, 0})
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

    VulkanUnorderedAccessView::VulkanUnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format)
        : resource_(&texture_array), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE), buff_range_({0, 0})
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

    VulkanUnorderedAccessView::VulkanUnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format)
        : resource_(&texture), sub_resource_(sub_resource), image_view_(VulkanImp(gpu_system), VK_NULL_HANDLE), buff_range_({0, 0})
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

    VulkanUnorderedAccessView::VulkanUnorderedAccessView(
        GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format)
        : resource_(&buffer), sub_resource_(0), buff_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
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
            const VkBufferViewCreateInfo view_create_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
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

    VulkanUnorderedAccessView::VulkanUnorderedAccessView(
        GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size)
        : resource_(&buffer), sub_resource_(0), buff_view_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkBuffer vulkan_buff = VulkanImp(buffer).Buffer();

        buff_range_ = {first_element * element_size, num_elements * element_size};

        this->FillWriteDescSetForStructuredBuffer(vulkan_buff);
    }

    VulkanUnorderedAccessView::~VulkanUnorderedAccessView()
    {
        this->Reset();
    }

    VulkanUnorderedAccessView::VulkanUnorderedAccessView(VulkanUnorderedAccessView&& other) noexcept = default;
    VulkanUnorderedAccessView::VulkanUnorderedAccessView(GpuUnorderedAccessViewInternal&& other) noexcept
        : VulkanUnorderedAccessView(static_cast<VulkanUnorderedAccessView&&>(other))
    {
    }
    VulkanUnorderedAccessView& VulkanUnorderedAccessView::operator=(VulkanUnorderedAccessView&& other) noexcept = default;
    GpuUnorderedAccessViewInternal& VulkanUnorderedAccessView::operator=(GpuUnorderedAccessViewInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanUnorderedAccessView&&>(other));
    }

    void VulkanUnorderedAccessView::Reset()
    {
        buff_view_.Reset();
        image_view_.Reset();
    }

    void VulkanUnorderedAccessView::Transition(GpuCommandList& cmd_list) const
    {
        this->Transition(VulkanImp(cmd_list));
    }

    void VulkanUnorderedAccessView::Transition(VulkanCommandList& cmd_list) const
    {
        VulkanImp(*resource_).Transition(cmd_list, sub_resource_, GpuResourceState::UnorderedAccess);
    }

    GpuResource* VulkanUnorderedAccessView::Resource() noexcept
    {
        return resource_;
    }

    VkWriteDescriptorSet VulkanUnorderedAccessView::WriteDescSet() const noexcept
    {
        return write_desc_set_;
    }

    VkImage VulkanUnorderedAccessView::Image() const noexcept
    {
        if (image_view_)
        {
            return static_cast<VulkanTexture&>(VulkanImp(*resource_)).Image();
        }
        else
        {
            return VK_NULL_HANDLE;
        }
    }
    VkImageView VulkanUnorderedAccessView::ImageView() const noexcept
    {
        return image_view_.Object();
    }
    const VkImageSubresourceRange& VulkanUnorderedAccessView::SubresourceRange() const
    {
        return subres_range_;
    }

    VkBuffer VulkanUnorderedAccessView::Buffer() const noexcept
    {
        if (buff_view_)
        {
            return static_cast<VulkanBuffer&>(VulkanImp(*resource_)).Buffer();
        }
        else
        {
            return VK_NULL_HANDLE;
        }
    }
    VkBufferView VulkanUnorderedAccessView::BufferView() const noexcept
    {
        return buff_view_.Object();
    }
    const std::tuple<uint32_t, uint32_t>& VulkanUnorderedAccessView::BufferRange() const
    {
        return buff_range_;
    }

    void VulkanUnorderedAccessView::CreateImageView(VkDevice device, VkImage image, VkImageViewType type, GpuFormat format)
    {
        const VkImageViewCreateInfo view_create_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = type,
            .format = ToVkFormat(format),
            .subresourceRange = subres_range_,
        };
        TIFVK(vkCreateImageView(device, &view_create_info, nullptr, &image_view_.Object()));
    }

    void VulkanUnorderedAccessView::FillWriteDescSetForImage()
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

    void VulkanUnorderedAccessView::FillWriteDescSetForStructuredBuffer(VkBuffer buff)
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
