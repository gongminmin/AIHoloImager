// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"

#include "../GpuResourceViewsInternal.hpp"
#include "VulkanCommandList.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    class VulkanShaderResourceView : public GpuShaderResourceViewInternal
    {
    public:
        VulkanShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format);
        VulkanShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format);
        VulkanShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format);

        VulkanShaderResourceView(
            GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format);
        VulkanShaderResourceView(
            GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);

        ~VulkanShaderResourceView() override;

        VulkanShaderResourceView(VulkanShaderResourceView&& other) noexcept;
        explicit VulkanShaderResourceView(GpuShaderResourceViewInternal&& other) noexcept;
        VulkanShaderResourceView& operator=(VulkanShaderResourceView&& other) noexcept;
        GpuShaderResourceViewInternal& operator=(GpuShaderResourceViewInternal&& other) noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list) const override;
        void Transition(VulkanCommandList& cmd_list) const;

        VkWriteDescriptorSet WriteDescSet() const noexcept;

    private:
        void FillWriteDescSetForImage();
        void FillWriteDescSetForStructuredBuffer(VkBuffer buff, uint32_t offset, uint32_t range);

    private:
        const GpuResource* resource_ = nullptr;
        uint32_t sub_resource_;
        VkWriteDescriptorSet write_desc_set_;

        VkDescriptorImageInfo image_info_{};
        VulkanRecyclableObject<VkImageView> image_view_;

        VkDescriptorBufferInfo buff_info_{};
        VulkanRecyclableObject<VkBufferView> buff_view_;
    };

    VULKAN_DEFINE_IMP(ShaderResourceView)

    class VulkanRenderTargetView : public GpuRenderTargetViewInternal
    {
    public:
        VulkanRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format);
        ~VulkanRenderTargetView() override;

        VulkanRenderTargetView(VulkanRenderTargetView&& other) noexcept;
        explicit VulkanRenderTargetView(GpuRenderTargetViewInternal&& other) noexcept;
        VulkanRenderTargetView& operator=(VulkanRenderTargetView&& other) noexcept;
        GpuRenderTargetViewInternal& operator=(GpuRenderTargetViewInternal&& other) noexcept override;

        explicit operator bool() const noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list) const override;
        void Transition(VulkanCommandList& cmd_list) const;

        GpuResource* Resource() const noexcept;
        VkImageView ImageView() const noexcept;
        VkImage Image() const noexcept;
        const VkImageSubresourceRange& SubresourceRange() const;

    private:
        GpuResource* resource_ = nullptr;
        uint32_t sub_resource_;
        VulkanRecyclableObject<VkImageView> image_view_;
        VkImageSubresourceRange subres_range_{};
    };

    VULKAN_DEFINE_IMP(RenderTargetView)

    class VulkanDepthStencilView : public GpuDepthStencilViewInternal
    {
    public:
        VulkanDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format);
        ~VulkanDepthStencilView() override;

        VulkanDepthStencilView(VulkanDepthStencilView&& other) noexcept;
        explicit VulkanDepthStencilView(GpuDepthStencilViewInternal&& other) noexcept;
        VulkanDepthStencilView& operator=(VulkanDepthStencilView&& other) noexcept;
        GpuDepthStencilViewInternal& operator=(GpuDepthStencilViewInternal&& other) noexcept override;

        explicit operator bool() const noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list) const override;
        void Transition(VulkanCommandList& cmd_list) const;

        GpuResource* Resource() const noexcept;
        VkImageView ImageView() const noexcept;
        VkImage Image() const noexcept;
        const VkImageSubresourceRange& SubresourceRange() const;

    private:
        GpuResource* resource_ = nullptr;
        uint32_t sub_resource_;
        VulkanRecyclableObject<VkImageView> image_view_;
        VkImageSubresourceRange subres_range_{};
    };

    VULKAN_DEFINE_IMP(DepthStencilView)

    class VulkanUnorderedAccessView : public GpuUnorderedAccessViewInternal
    {
    public:
        VulkanUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format);
        VulkanUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format);
        VulkanUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format);

        VulkanUnorderedAccessView(
            GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format);

        VulkanUnorderedAccessView(
            GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);

        ~VulkanUnorderedAccessView() override;

        VulkanUnorderedAccessView(VulkanUnorderedAccessView&& other) noexcept;
        explicit VulkanUnorderedAccessView(GpuUnorderedAccessViewInternal&& other) noexcept;
        VulkanUnorderedAccessView& operator=(VulkanUnorderedAccessView&& other) noexcept;
        GpuUnorderedAccessViewInternal& operator=(GpuUnorderedAccessViewInternal&& other) noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list) const override;
        void Transition(VulkanCommandList& cmd_list) const;

        GpuResource* Resource() noexcept override;
        VkWriteDescriptorSet WriteDescSet() const noexcept;

        VkImage Image() const noexcept;
        VkImageView ImageView() const noexcept;
        const VkImageSubresourceRange& SubresourceRange() const;

        VkBuffer Buffer() const noexcept;
        VkBufferView BufferView() const noexcept;
        const std::tuple<uint32_t, uint32_t>& BufferRange() const;

    private:
        void CreateImageView(VkDevice device, VkImage image, VkImageViewType type, GpuFormat format);
        void FillWriteDescSetForImage();
        void FillWriteDescSetForStructuredBuffer(VkBuffer buff);

    private:
        GpuResource* resource_ = nullptr;
        uint32_t sub_resource_;
        VkWriteDescriptorSet write_desc_set_;

        VkDescriptorImageInfo image_info_{};
        VkImageSubresourceRange subres_range_{};
        VulkanRecyclableObject<VkImageView> image_view_;

        VkDescriptorBufferInfo buffer_info_{};
        std::tuple<uint32_t, uint32_t> buff_range_{};
        VulkanRecyclableObject<VkBufferView> buff_view_;
    };

    VULKAN_DEFINE_IMP(UnorderedAccessView)

    VkWriteDescriptorSet NullSampledImageShaderResourceViewWriteDescSet();
    VkWriteDescriptorSet NullUniformTexelBufferShaderResourceViewWriteDescSet();

    VkWriteDescriptorSet NullStorageImageUnorderedAccessViewWriteDescSet();
    VkWriteDescriptorSet NullStorageTexelBufferUnorderedAccessViewWriteDescSet();
    VkWriteDescriptorSet NullStorageBufferUnorderedAccessViewWriteDescSet();

    VkWriteDescriptorSet NullUniformBufferConstantBufferViewWriteDescSet();
} // namespace AIHoloImager
