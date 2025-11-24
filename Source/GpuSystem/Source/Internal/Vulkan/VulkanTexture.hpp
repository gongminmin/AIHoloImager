// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <volk.h>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuTexture.hpp"

#include "../GpuTextureInternal.hpp"
#include "VulkanCommandList.hpp"
#include "VulkanImpDefine.hpp"
#include "VulkanResource.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class VulkanTexture : public GpuTextureInternal, public VulkanResource
    {
    public:
        VulkanTexture(GpuSystem& gpu_system, GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size,
            uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::string_view name);
        VulkanTexture(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::string_view name);
        ~VulkanTexture() noexcept;

        VulkanTexture(VulkanTexture&& other) noexcept;
        explicit VulkanTexture(GpuResourceInternal&& other) noexcept;
        explicit VulkanTexture(GpuTextureInternal&& other) noexcept;
        VulkanTexture& operator=(VulkanTexture&& other) noexcept;
        GpuResourceInternal& operator=(GpuResourceInternal&& other) noexcept override;
        GpuTextureInternal& operator=(GpuTextureInternal&& other) noexcept override;

        void Name(std::string_view name) override;

        VkImage Image() const noexcept;
        void* NativeResource() const noexcept override;
        void* NativeTexture() const noexcept override;

        void* SharedHandle() const noexcept override;

        GpuResourceType Type() const noexcept override;
        GpuResourceFlag Flags() const noexcept override;
        uint32_t AllocationSize() const noexcept override;

        uint32_t Width(uint32_t mip) const noexcept override;
        uint32_t Height(uint32_t mip) const noexcept override;
        uint32_t Depth(uint32_t mip) const noexcept override;
        uint32_t ArraySize() const noexcept override;
        uint32_t MipLevels() const noexcept override;
        uint32_t Planes() const noexcept override;
        GpuFormat Format() const noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const override;
        void Transition(VulkanCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        void Transition(VulkanCommandList& cmd_list, GpuResourceState target_state) const override;

    private:
        mutable std::vector<VkImageLayout> curr_layouts_;
        GpuFormat format_{};
        GpuResourceFlag flags_{};

        VulkanRecyclableObject<VkImage> texture_;
        VkImageCreateInfo image_create_info_;
    };

    VULKAN_DEFINE_IMP(Texture)
} // namespace AIHoloImager
