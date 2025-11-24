// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuResource.hpp"

#include "VulkanCommandList.hpp"
#include "VulkanImpDefine.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class VulkanResource
    {
    public:
        explicit VulkanResource(GpuSystem& gpu_system);
        virtual ~VulkanResource();

        VulkanResource(VulkanResource&& other) noexcept;
        VulkanResource& operator=(VulkanResource&& other) noexcept;

        virtual void Transition(VulkanCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const = 0;
        virtual void Transition(VulkanCommandList& cmd_list, GpuResourceState target_state) const = 0;

    protected:
        void CreateMemory(GpuResourceType type, const VkMemoryRequirements& requirements, GpuHeap heap, GpuResourceFlag flags);
        void Name(void* object, std::string_view name);
        void Reset();
        VkDeviceMemory Memory() const noexcept;
        void* SharedHandle() const noexcept;
        GpuResourceType Type() const noexcept;
        GpuResourceFlag Flags() const noexcept;

    private:
        GpuResourceType type_ = GpuResourceType::Buffer;
        GpuResourceFlag flags_ = GpuResourceFlag::None;
        VulkanRecyclableObject<VkDeviceMemory> memory_;
        Win32UniqueHandle shared_handle_;
    };

    VULKAN_DEFINE_IMP(Resource)
} // namespace AIHoloImager
