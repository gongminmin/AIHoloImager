// Copyright (c) 2025 Minmin Gong
//

#include "VulkanBuffer.hpp"

#include <cassert>

#include "VulkanCommandList.hpp"
#include "VulkanConversion.hpp"
#include "VulkanErrorhandling.hpp"
#include "VulkanSystem.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(Buffer)

    VulkanBuffer::VulkanBuffer(GpuSystem& gpu_system, uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name)
        : VulkanResource(gpu_system), heap_(heap),
          curr_state_(heap == GpuHeap::ReadBack ? GpuResourceState::CopyDst : GpuResourceState::Common),
          buff_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const auto& vulkan_system = *buff_.VulkanSys();
        const VkDevice vulkan_device = vulkan_system.Device();

        VkBufferUsageFlags usage = 0;
        switch (heap)
        {
        case GpuHeap::Default:
            usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
            if (EnumHasAny(flags, GpuResourceFlag::UnorderedAccess))
            {
                usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                if (!EnumHasAny(flags, GpuResourceFlag::Structured))
                {
                    usage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
                }
            }
            if (!EnumHasAny(flags, GpuResourceFlag::Structured))
            {
                usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
            }
            break;

        case GpuHeap::Upload:
            usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            break;

        case GpuHeap::ReadBack:
            usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            break;
        }

        buff_create_info_ = VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        VkExternalMemoryBufferCreateInfo external_mem_buff_create_info;
        if (EnumHasAny(flags, GpuResourceFlag::Shareable))
        {
            external_mem_buff_create_info = {
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            };

            buff_create_info_.pNext = &external_mem_buff_create_info;
        }

        TIFVK(vkCreateBuffer(vulkan_device, &buff_create_info_, nullptr, &buff_.Object()));

        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(vulkan_device, buff_.Object(), &requirements);
        this->CreateMemory(GpuResourceType::Buffer, requirements, heap, flags);
        TIFVK(vkBindBufferMemory(vulkan_device, buff_.Object(), this->Memory(), 0));

        this->Name(std::move(name));
    }
    VulkanBuffer::VulkanBuffer(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::string_view name)
        : VulkanResource(gpu_system), buff_(VulkanImp(gpu_system), reinterpret_cast<VkBuffer>(native_resource)), curr_state_(curr_state)
    {
        if (buff_.Object() != VK_NULL_HANDLE)
        {
            this->Name(std::move(name));

            /*D3D12_HEAP_PROPERTIES heap_prop;
            static_cast<ID3D12Resource*>(native_resource)->GetHeapProperties(&heap_prop, nullptr);
            heap_ = FromD3D12HeapType(heap_prop.Type);*/
        }
    }

    VulkanBuffer::~VulkanBuffer() = default;

    VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept = default;
    VulkanBuffer::VulkanBuffer(GpuResourceInternal&& other) noexcept : VulkanBuffer(static_cast<VulkanBuffer&&>(other))
    {
    }
    VulkanBuffer::VulkanBuffer(GpuBufferInternal&& other) noexcept : VulkanBuffer(static_cast<VulkanBuffer&&>(other))
    {
    }
    VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept = default;
    GpuResourceInternal& VulkanBuffer::operator=(GpuResourceInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanBuffer&&>(other));
    }
    GpuBufferInternal& VulkanBuffer::operator=(GpuBufferInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanBuffer&&>(other));
    }

    void VulkanBuffer::Name(std::string_view name)
    {
        this->VulkanResource::Name(buff_.Object(), std::move(name));
    }

    VkBuffer VulkanBuffer::Buffer() const noexcept
    {
        return buff_.Object();
    }

    void* VulkanBuffer::NativeResource() const noexcept
    {
        return this->Buffer();
    }

    void* VulkanBuffer::NativeBuffer() const noexcept
    {
        return this->NativeResource();
    }

    void* VulkanBuffer::SharedHandle() const noexcept
    {
        return this->VulkanResource::SharedHandle();
    }

    GpuHeap VulkanBuffer::Heap() const noexcept
    {
        return heap_;
    }

    GpuResourceType VulkanBuffer::Type() const noexcept
    {
        return this->VulkanResource::Type();
    }

    uint32_t VulkanBuffer::AllocationSize() const noexcept
    {
        const VkDevice vulkan_device = buff_.VulkanSys()->Device();

        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(vulkan_device, buff_.Object(), &requirements);
        return static_cast<uint32_t>(requirements.size);
    }

    GpuResourceFlag VulkanBuffer::Flags() const noexcept
    {
        return this->VulkanResource::Flags();
    }

    uint32_t VulkanBuffer::Size() const noexcept
    {
        return static_cast<uint32_t>(buff_create_info_.size);
    }

    void* VulkanBuffer::Map(const GpuRange& read_range)
    {
        const VkDevice vulkan_device = buff_.VulkanSys()->Device();

        void* addr;
        TIFVK(vkMapMemory(vulkan_device, this->Memory(), read_range.begin, read_range.end - read_range.begin, 0, &addr));

        VkMappedMemoryRange vulkan_read_range{
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = this->Memory(),
        };
        if (heap_ == GpuHeap::ReadBack)
        {
            vulkan_read_range.offset = 0;
            vulkan_read_range.size = VK_WHOLE_SIZE;
        }
        else
        {
            vulkan_read_range.offset = read_range.begin;
            vulkan_read_range.size = read_range.end - read_range.begin;
        }
        vkInvalidateMappedMemoryRanges(vulkan_device, 1, &vulkan_read_range);

        return addr;
    }

    void* VulkanBuffer::Map()
    {
        return this->Map(GpuRange{0, VK_WHOLE_SIZE});
    }

    void VulkanBuffer::Unmap(const GpuRange& write_range)
    {
        const VkDevice vulkan_device = buff_.VulkanSys()->Device();

        VkMappedMemoryRange vulkan_write_range{
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = this->Memory(),
        };
        if (heap_ == GpuHeap::Upload)
        {
            vulkan_write_range.offset = 0;
            vulkan_write_range.size = VK_WHOLE_SIZE;
        }
        else
        {
            vulkan_write_range.offset = write_range.begin;
            vulkan_write_range.size = write_range.end - write_range.begin;
        }
        vkFlushMappedMemoryRanges(vulkan_device, 1, &vulkan_write_range);

        vkUnmapMemory(vulkan_device, this->Memory());
    }

    void VulkanBuffer::Unmap()
    {
        this->Unmap(GpuRange{0, VK_WHOLE_SIZE});
    }

    void VulkanBuffer::Reset()
    {
        this->VulkanResource::Reset();

        buff_.Reset();
        buff_create_info_ = {};
        heap_ = {};
        curr_state_ = {};
    }

    void VulkanBuffer::Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const
    {
        this->Transition(VulkanImp(cmd_list), sub_resource, target_state);
    }

    void VulkanBuffer::Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const
    {
        this->Transition(VulkanImp(cmd_list), target_state);
    }

    void VulkanBuffer::Transition(VulkanCommandList& cmd_list, [[maybe_unused]] uint32_t sub_resource, GpuResourceState target_state) const
    {
        assert(sub_resource == 0);
        this->Transition(cmd_list, target_state);
    }

    void VulkanBuffer::Transition(VulkanCommandList& cmd_list, GpuResourceState target_state) const
    {
        if ((curr_state_ != target_state) || (target_state == GpuResourceState::UnorderedAccess) ||
            (target_state == GpuResourceState::RayTracingAS))
        {
            const VkBufferMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = buff_.Object(),
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            };
            cmd_list.Transition(std::span(&barrier, 1));

            curr_state_ = target_state;
        }
    }
} // namespace AIHoloImager
