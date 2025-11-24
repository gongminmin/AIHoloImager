// Copyright (c) 2025 Minmin Gong
//

#include "VulkanConversion.hpp"

#include "Base/ErrorHandling.hpp"

namespace AIHoloImager
{
    VkFormat ToVkFormat(GpuFormat fmt)
    {
        switch (fmt)
        {
        case GpuFormat::Unknown:
            return VK_FORMAT_UNDEFINED;

        case GpuFormat::R8_UNorm:
            return VK_FORMAT_R8_UNORM;

        case GpuFormat::RG8_UNorm:
            return VK_FORMAT_R8G8_UNORM;

        case GpuFormat::RGBA8_UNorm:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case GpuFormat::RGBA8_UNorm_SRGB:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case GpuFormat::BGRA8_UNorm:
            return VK_FORMAT_B8G8R8A8_UNORM;
        case GpuFormat::BGRA8_UNorm_SRGB:
            return VK_FORMAT_B8G8R8A8_SRGB;
        case GpuFormat::BGRX8_UNorm:
            return VK_FORMAT_B8G8R8A8_UNORM;
        case GpuFormat::BGRX8_UNorm_SRGB:
            return VK_FORMAT_B8G8R8A8_SRGB;

        case GpuFormat::R16_Uint:
            return VK_FORMAT_R16_UINT;
        case GpuFormat::R16_Sint:
            return VK_FORMAT_R16_SINT;
        case GpuFormat::R16_Float:
            return VK_FORMAT_R16_SFLOAT;

        case GpuFormat::RG16_Uint:
            return VK_FORMAT_R16G16_UINT;
        case GpuFormat::RG16_Sint:
            return VK_FORMAT_R16G16_SINT;
        case GpuFormat::RG16_Float:
            return VK_FORMAT_R16G16_SFLOAT;

        case GpuFormat::RGBA16_Uint:
            return VK_FORMAT_R16G16B16A16_UINT;
        case GpuFormat::RGBA16_Sint:
            return VK_FORMAT_R16G16B16A16_SINT;
        case GpuFormat::RGBA16_Float:
            return VK_FORMAT_R16G16B16A16_SFLOAT;

        case GpuFormat::R32_Uint:
            return VK_FORMAT_R32_UINT;
        case GpuFormat::R32_Sint:
            return VK_FORMAT_R32_SINT;
        case GpuFormat::R32_Float:
            return VK_FORMAT_R32_SFLOAT;

        case GpuFormat::RG32_Uint:
            return VK_FORMAT_R32G32_UINT;
        case GpuFormat::RG32_Sint:
            return VK_FORMAT_R32G32_SINT;
        case GpuFormat::RG32_Float:
            return VK_FORMAT_R32G32_SFLOAT;

        case GpuFormat::RGB32_Uint:
            return VK_FORMAT_R32G32B32_UINT;
        case GpuFormat::RGB32_Sint:
            return VK_FORMAT_R32G32B32_SINT;
        case GpuFormat::RGB32_Float:
            return VK_FORMAT_R32G32B32_SFLOAT;

        case GpuFormat::RGBA32_Uint:
            return VK_FORMAT_R32G32B32A32_UINT;
        case GpuFormat::RGBA32_Sint:
            return VK_FORMAT_R32G32B32A32_SINT;
        case GpuFormat::RGBA32_Float:
            return VK_FORMAT_R32G32B32A32_SFLOAT;

        case GpuFormat::D16_UNorm:
            return VK_FORMAT_D16_UNORM;
        case GpuFormat::D24_UNorm_S8_Uint:
            return VK_FORMAT_D24_UNORM_S8_UINT;
        case GpuFormat::D32_Float:
            return VK_FORMAT_D32_SFLOAT;
        case GpuFormat::D32_Float_S8X24_Uint:
            return VK_FORMAT_D32_SFLOAT_S8_UINT;

        default:
            Unreachable("Invalid format");
        }
    }

    GpuFormat FromVkFormat(VkFormat fmt)
    {
        switch (fmt)
        {
        case VK_FORMAT_UNDEFINED:
            return GpuFormat::Unknown;

        case VK_FORMAT_R8_UNORM:
            return GpuFormat::R8_UNorm;

        case VK_FORMAT_R8G8_UNORM:
            return GpuFormat::RG8_UNorm;

        case VK_FORMAT_R8G8B8A8_UNORM:
            return GpuFormat::RGBA8_UNorm;
        case VK_FORMAT_R8G8B8A8_SRGB:
            return GpuFormat::RGBA8_UNorm_SRGB;
        case VK_FORMAT_B8G8R8A8_UNORM:
            return GpuFormat::BGRA8_UNorm;
        case VK_FORMAT_B8G8R8A8_SRGB:
            return GpuFormat::BGRA8_UNorm_SRGB;

        case VK_FORMAT_R16_UINT:
            return GpuFormat::R16_Uint;
        case VK_FORMAT_R16_SINT:
            return GpuFormat::R16_Sint;
        case VK_FORMAT_R16_SFLOAT:
            return GpuFormat::R16_Float;

        case VK_FORMAT_R16G16_UINT:
            return GpuFormat::RG16_Uint;
        case VK_FORMAT_R16G16_SINT:
            return GpuFormat::RG16_Sint;
        case VK_FORMAT_R16G16_SFLOAT:
            return GpuFormat::RG16_Float;

        case VK_FORMAT_R16G16B16A16_UINT:
            return GpuFormat::RGBA16_Uint;
        case VK_FORMAT_R16G16B16A16_SINT:
            return GpuFormat::RGBA16_Sint;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return GpuFormat::RGBA16_Float;

        case VK_FORMAT_R32_UINT:
            return GpuFormat::R32_Uint;
        case VK_FORMAT_R32_SINT:
            return GpuFormat::R32_Sint;
        case VK_FORMAT_R32_SFLOAT:
            return GpuFormat::R32_Float;

        case VK_FORMAT_R32G32_UINT:
            return GpuFormat::RG32_Uint;
        case VK_FORMAT_R32G32_SINT:
            return GpuFormat::RG32_Sint;
        case VK_FORMAT_R32G32_SFLOAT:
            return GpuFormat::RG32_Float;

        case VK_FORMAT_R32G32B32_UINT:
            return GpuFormat::RGB32_Uint;
        case VK_FORMAT_R32G32B32_SINT:
            return GpuFormat::RGB32_Sint;
        case VK_FORMAT_R32G32B32_SFLOAT:
            return GpuFormat::RGB32_Float;

        case VK_FORMAT_R32G32B32A32_UINT:
            return GpuFormat::RGBA32_Uint;
        case VK_FORMAT_R32G32B32A32_SINT:
            return GpuFormat::RGBA32_Sint;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return GpuFormat::RGBA32_Float;

        case VK_FORMAT_D16_UNORM:
            return GpuFormat::D16_UNorm;
        case VK_FORMAT_D24_UNORM_S8_UINT:
            return GpuFormat::D24_UNorm_S8_Uint;
        case VK_FORMAT_D32_SFLOAT:
            return GpuFormat::D32_Float;
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return GpuFormat::D32_Float_S8X24_Uint;

        default:
            Unreachable("Unsupported Vulkan format");
        }
    }

    VkMemoryPropertyFlags ToVulkanMemoryPropertyFlags(GpuHeap heap)
    {
        switch (heap)
        {
        case GpuHeap::Default:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        case GpuHeap::Upload:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        case GpuHeap::ReadBack:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

        default:
            Unreachable("Invalid heap type");
        }
    }

    GpuHeap FromVulkanMemoryPropertyFlags(VkMemoryPropertyFlags flags)
    {
        switch (flags)
        {
        case VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:
            return GpuHeap::Default;
        case VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:
            return GpuHeap::Upload;
        case VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT:
            return GpuHeap::ReadBack;
        default:
            Unreachable("Unsupported Vulkan memory property flags");
        }
    }

    VkImageUsageFlags ToVulkanImageUsageFlags(GpuResourceFlag flags) noexcept
    {
        VkImageUsageFlags vulkan_flag = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (EnumHasAny(flags, GpuResourceFlag::RenderTarget))
        {
            vulkan_flag |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
        if (EnumHasAny(flags, GpuResourceFlag::DepthStencil))
        {
            vulkan_flag |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        }
        if (EnumHasAny(flags, GpuResourceFlag::UnorderedAccess))
        {
            vulkan_flag |= VK_IMAGE_USAGE_STORAGE_BIT;
        }

        return vulkan_flag;
    }

    GpuResourceFlag FromVulkanImageUsageFlags(VkImageUsageFlags flags) noexcept
    {
        GpuResourceFlag gpu_flag = GpuResourceFlag::None;
        if (flags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        {
            gpu_flag |= GpuResourceFlag::RenderTarget;
        }
        if (flags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        {
            gpu_flag |= GpuResourceFlag::DepthStencil;
        }
        if (flags & VK_IMAGE_USAGE_STORAGE_BIT)
        {
            gpu_flag |= GpuResourceFlag::UnorderedAccess;
        }
        return gpu_flag;
    }

    VkImageLayout ToVulkanImageLayout(GpuResourceState state)
    {
        switch (state)
        {
        case GpuResourceState::Common:
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        case GpuResourceState::ColorWrite:
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        case GpuResourceState::DepthWrite:
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        case GpuResourceState::UnorderedAccess:
            return VK_IMAGE_LAYOUT_GENERAL;

        case GpuResourceState::CopySrc:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case GpuResourceState::CopyDst:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        default:
            Unreachable("Invalid image layout");
        }
    }

    VkImageType ToVulkanImageType(GpuResourceType type)
    {
        switch (type)
        {
        case GpuResourceType::Texture2D:
        case GpuResourceType::Texture2DArray:
            return VK_IMAGE_TYPE_2D;
        case GpuResourceType::Texture3D:
            return VK_IMAGE_TYPE_3D;
        default:
            Unreachable("Invalid image type");
        }
    }

    std::tuple<VkAccessFlags, VkAccessFlags> ToVulkanAccessFlags(VkImageLayout old_layout, VkImageLayout new_layout)
    {
        VkAccessFlags src_access_mask;
        switch (old_layout)
        {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            src_access_mask = 0;
            break;
        case VK_IMAGE_LAYOUT_PREINITIALIZED:
            src_access_mask = VK_ACCESS_HOST_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            src_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            src_access_mask = VK_ACCESS_TRANSFER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            src_access_mask = VK_ACCESS_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_GENERAL:
            src_access_mask = VK_ACCESS_SHADER_WRITE_BIT;
            break;

        default:
            Unreachable("Unsupported old layout");
        }

        VkAccessFlags dst_access_mask;
        switch (new_layout)
        {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            dst_access_mask = VK_ACCESS_TRANSFER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            dst_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            if (src_access_mask == 0)
            {
                src_access_mask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_GENERAL:
            dst_access_mask = VK_ACCESS_SHADER_WRITE_BIT;
            break;

        default:
            Unreachable("Unsupported new layout");
        }

        return {src_access_mask, dst_access_mask};
    }

    VkImageAspectFlags ToVulkanAspectMask(GpuFormat fmt)
    {
        VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_NONE;
        if (IsDepthStencilFormat(fmt))
        {
            aspect_mask |= VK_IMAGE_ASPECT_DEPTH_BIT;
            if (IsStencilFormat(fmt))
            {
                aspect_mask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }
        else
        {
            aspect_mask |= VK_IMAGE_ASPECT_COLOR_BIT;
        }
        return aspect_mask;
    }
} // namespace AIHoloImager
