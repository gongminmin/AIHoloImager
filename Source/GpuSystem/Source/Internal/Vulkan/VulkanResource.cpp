// Copyright (c) 2025 Minmin Gong
//

#include "VulkanResource.hpp"

#include "VulkanBuffer.hpp"
#include "VulkanConversion.hpp"
#include "VulkanErrorHandling.hpp"
#include "VulkanSystem.hpp"
#include "VulkanTexture.hpp"

namespace AIHoloImager
{
    VulkanResource::VulkanResource(GpuSystem& gpu_system) : memory_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
    }

    VulkanResource::~VulkanResource()
    {
        this->Reset();
    }

    VulkanResource::VulkanResource(VulkanResource&& other) noexcept = default;

    VulkanResource& VulkanResource::operator=(VulkanResource&& other) noexcept = default;

    void VulkanResource::CreateMemory(GpuResourceType type, const VkMemoryRequirements& requirements, GpuHeap heap, GpuResourceFlag flags)
    {
        type_ = type;
        flags_ = flags;

        const auto& vulkan_system = *memory_.VulkanSys();
        const VkDevice vulkan_device = vulkan_system.Device();

        VkMemoryAllocateInfo mem_alloc_info{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex = vulkan_system.MemoryTypeIndex(requirements.memoryTypeBits, ToVulkanMemoryPropertyFlags(heap)),
        };

        VkExportMemoryAllocateInfo export_mem_alloc_info;
        VkExportMemoryWin32HandleInfoKHR export_mem_win32_handle_info;
        if (EnumHasAny(flags, GpuResourceFlag::Shareable))
        {
            export_mem_win32_handle_info = {
                .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
                .pAttributes = nullptr,
                .dwAccess = GENERIC_ALL,
                .name = nullptr,
            };
            export_mem_alloc_info = {
                .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                .pNext = &export_mem_win32_handle_info,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };

            mem_alloc_info.pNext = &export_mem_alloc_info;
        }

        TIFVK(vkAllocateMemory(vulkan_device, &mem_alloc_info, nullptr, &memory_.Object()));

        if (EnumHasAny(flags, GpuResourceFlag::Shareable))
        {
            const VkMemoryGetWin32HandleInfoKHR get_win32_handle_info{
                .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
                .memory = memory_.Object(),
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };

            HANDLE shared_handle;
            TIFVK(vkGetMemoryWin32HandleKHR(vulkan_device, &get_win32_handle_info, &shared_handle));
            shared_handle_.reset(shared_handle);
        }
    }

    void VulkanResource::Name(void* object, std::string_view name)
    {
        SetName(
            *memory_.VulkanSys(), object, type_ == GpuResourceType::Buffer ? VK_OBJECT_TYPE_BUFFER : VK_OBJECT_TYPE_IMAGE, std::move(name));
    }

    void VulkanResource::Reset()
    {
        memory_.Reset();
    }

    VkDeviceMemory VulkanResource::Memory() const noexcept
    {
        return memory_.Object();
    }

    void* VulkanResource::SharedHandle() const noexcept
    {
        return shared_handle_.get();
    }

    GpuResourceType VulkanResource::Type() const noexcept
    {
        return type_;
    }

    GpuResourceFlag VulkanResource::Flags() const noexcept
    {
        return flags_;
    }

    VulkanResource& VulkanImp(GpuResource& resource)
    {
        if (resource.Type() == GpuResourceType::Buffer)
        {
            return static_cast<VulkanBuffer&>(resource.Internal());
        }
        else
        {
            return static_cast<VulkanTexture&>(resource.Internal());
        }
    }

    const VulkanResource& VulkanImp(const GpuResource& resource)
    {
        return VulkanImp(const_cast<GpuResource&>(resource));
    }
} // namespace AIHoloImager
