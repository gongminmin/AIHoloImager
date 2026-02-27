// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include <volk.h>

#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuCommandList.hpp"

#include "../GpuBufferInternal.hpp"
#include "VulkanCommandList.hpp"
#include "VulkanImpDefine.hpp"
#include "VulkanResource.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class VulkanBuffer : public GpuBufferInternal, public VulkanResource
    {
    public:
        VulkanBuffer(GpuSystem& gpu_system, uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name);
        ~VulkanBuffer() override;

        VulkanBuffer(VulkanBuffer&& other) noexcept;
        explicit VulkanBuffer(GpuResourceInternal&& other) noexcept;
        explicit VulkanBuffer(GpuBufferInternal&& other) noexcept;
        VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;
        GpuResourceInternal& operator=(GpuResourceInternal&& other) noexcept override;
        GpuBufferInternal& operator=(GpuBufferInternal&& other) noexcept override;

        void Name(std::string_view name) override;

        VkBuffer Buffer() const noexcept;
        void* NativeResource() const noexcept override;
        void* NativeBuffer() const noexcept override;

        void* SharedHandle() const noexcept override;

        GpuHeap Heap() const noexcept override;
        GpuResourceType Type() const noexcept override;
        GpuResourceFlag Flags() const noexcept override;
        uint32_t AllocationSize() const noexcept override;

        uint32_t Size() const noexcept override;

        void* Map(const GpuRange& read_range) override;
        void* Map() override;
        void Unmap(const GpuRange& write_range) override;
        void Unmap() override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const override;
        using VulkanResource::Transition;

    private:
        void DoTransition(VulkanCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        void DoTransition(VulkanCommandList& cmd_list, GpuResourceState target_state) const override;

    private:
        GpuHeap heap_{};
        mutable GpuResourceState curr_state_{};

        VulkanRecyclableObject<VkBuffer> buff_;
        VkBufferCreateInfo buff_create_info_;
    };

    VULKAN_DEFINE_IMP(Buffer)
} // namespace AIHoloImager
