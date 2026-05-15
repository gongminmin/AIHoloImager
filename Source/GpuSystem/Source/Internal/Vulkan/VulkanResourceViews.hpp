// Copyright (c) 2025-2026 Minmin Gong
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
    class VulkanConstantBufferView : public GpuConstantBufferViewInternal
    {
    public:
        explicit VulkanConstantBufferView(const GpuMemoryBlock& mem_block);

        ~VulkanConstantBufferView() override;

        VulkanConstantBufferView(VulkanConstantBufferView&& other) noexcept;
        explicit VulkanConstantBufferView(GpuConstantBufferViewInternal&& other) noexcept;
        VulkanConstantBufferView& operator=(VulkanConstantBufferView&& other) noexcept;
        GpuConstantBufferViewInternal& operator=(GpuConstantBufferViewInternal&& other) noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list) const override;
        void Transition(VulkanCommandList& cmd_list) const;

        VkWriteDescriptorSet WriteDescSet() const noexcept;

    private:
        const GpuResource* resource_ = nullptr;
        const GpuMemoryBlock* mem_block_ = nullptr;
        VkWriteDescriptorSet write_desc_set_{};

        VkDescriptorBufferInfo buff_info_{};
        VulkanRecyclableObject<VkBufferView> buff_view_;
    };

    VULKAN_DEFINE_IMP(ConstantBufferView)

    class VulkanShaderResourceView : public GpuShaderResourceViewInternal
    {
    public:
        explicit VulkanShaderResourceView(const GpuResource& resource) noexcept;
        ~VulkanShaderResourceView() noexcept override;

        VulkanShaderResourceView(VulkanShaderResourceView&& other) noexcept;
        virtual VulkanShaderResourceView& operator=(VulkanShaderResourceView&& other) noexcept = 0;

        virtual bool IsBuffer() const noexcept = 0;

        void Transition(GpuCommandList& cmd_list) const override;
        virtual void Transition(VulkanCommandList& cmd_list) const = 0;

        const GpuResource* Resource() const noexcept;
        VkWriteDescriptorSet WriteDescSet() const noexcept;

    protected:
        const GpuResource* resource_ = nullptr;
        VkWriteDescriptorSet write_desc_set_{};
    };

    VULKAN_DEFINE_IMP(ShaderResourceView)

    class VulkanBufferShaderResourceView : public VulkanShaderResourceView
    {
    public:
        VulkanBufferShaderResourceView(
            GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format);
        VulkanBufferShaderResourceView(
            GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);

        ~VulkanBufferShaderResourceView() override;

        VulkanBufferShaderResourceView(VulkanBufferShaderResourceView&& other) noexcept;
        explicit VulkanBufferShaderResourceView(GpuShaderResourceViewInternal&& other) noexcept;
        explicit VulkanBufferShaderResourceView(VulkanShaderResourceView&& other) noexcept;
        VulkanBufferShaderResourceView& operator=(VulkanBufferShaderResourceView&& other) noexcept;
        GpuShaderResourceViewInternal& operator=(GpuShaderResourceViewInternal&& other) noexcept override;
        VulkanShaderResourceView& operator=(VulkanShaderResourceView&& other) noexcept override;

        bool IsBuffer() const noexcept override
        {
            return true;
        }

        void Reset() override;

        void Transition(VulkanCommandList& cmd_list) const override;

    private:
        void FillWriteDescSetForStructuredBuffer(VkBuffer buff, uint32_t offset, uint32_t range);

    private:
        VkDescriptorBufferInfo buff_info_{};
        VulkanRecyclableObject<VkBufferView> buff_view_;
    };

    VULKAN_DEFINE_IMP2(ShaderResourceView, BufferShaderResourceView)

    class VulkanTextureShaderResourceView : public VulkanShaderResourceView
    {
    public:
        VulkanTextureShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format);
        VulkanTextureShaderResourceView(
            GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format);
        VulkanTextureShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format);

        ~VulkanTextureShaderResourceView() override;

        VulkanTextureShaderResourceView(VulkanTextureShaderResourceView&& other) noexcept;
        explicit VulkanTextureShaderResourceView(GpuShaderResourceViewInternal&& other) noexcept;
        explicit VulkanTextureShaderResourceView(VulkanShaderResourceView&& other) noexcept;
        VulkanTextureShaderResourceView& operator=(VulkanTextureShaderResourceView&& other) noexcept;
        GpuShaderResourceViewInternal& operator=(GpuShaderResourceViewInternal&& other) noexcept override;
        VulkanShaderResourceView& operator=(VulkanShaderResourceView&& other) noexcept override;

        bool IsBuffer() const noexcept override
        {
            return false;
        }

        void Reset() override;

        void Transition(VulkanCommandList& cmd_list) const override;

    private:
        void FillWriteDescSetForImage();

    private:
        uint32_t sub_resource_;

        VkDescriptorImageInfo image_info_{};
        VulkanRecyclableObject<VkImageView> image_view_;
    };

    VULKAN_DEFINE_IMP2(ShaderResourceView, TextureShaderResourceView)

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
        void TransitionBack(VulkanCommandList& cmd_list) const;

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
        void TransitionBack(VulkanCommandList& cmd_list) const;

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
        explicit VulkanUnorderedAccessView(GpuResource& resource) noexcept;
        ~VulkanUnorderedAccessView() noexcept override;

        VulkanUnorderedAccessView(VulkanUnorderedAccessView&& other) noexcept;
        virtual VulkanUnorderedAccessView& operator=(VulkanUnorderedAccessView&& other) noexcept = 0;

        virtual bool IsBuffer() const noexcept = 0;

        void Transition(GpuCommandList& cmd_list) const override;
        virtual void Transition(VulkanCommandList& cmd_list) const = 0;

        GpuResource* Resource() noexcept override;
        VkWriteDescriptorSet WriteDescSet() const noexcept;

    protected:
        GpuResource* resource_ = nullptr;
        VkWriteDescriptorSet write_desc_set_{};
    };

    VULKAN_DEFINE_IMP(UnorderedAccessView)

    class VulkanBufferUnorderedAccessView : public VulkanUnorderedAccessView
    {
    public:
        VulkanBufferUnorderedAccessView(
            GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format);

        VulkanBufferUnorderedAccessView(
            GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);

        ~VulkanBufferUnorderedAccessView() override;

        VulkanBufferUnorderedAccessView(VulkanBufferUnorderedAccessView&& other) noexcept;
        explicit VulkanBufferUnorderedAccessView(GpuUnorderedAccessViewInternal&& other) noexcept;
        explicit VulkanBufferUnorderedAccessView(VulkanUnorderedAccessView&& other) noexcept;
        VulkanBufferUnorderedAccessView& operator=(VulkanBufferUnorderedAccessView&& other) noexcept;
        GpuUnorderedAccessViewInternal& operator=(GpuUnorderedAccessViewInternal&& other) noexcept override;
        VulkanUnorderedAccessView& operator=(VulkanUnorderedAccessView&& other) noexcept override;

        bool IsBuffer() const noexcept override
        {
            return true;
        }

        void Reset() override;

        void Transition(VulkanCommandList& cmd_list) const override;

        VkBuffer Buffer() const noexcept;
        VkBufferView BufferView() const noexcept;
        const std::tuple<uint32_t, uint32_t>& BufferRange() const;

    private:
        void FillWriteDescSetForStructuredBuffer(VkBuffer buff);

    private:
        VkDescriptorBufferInfo buffer_info_{};
        std::tuple<uint32_t, uint32_t> buff_range_{};
        VulkanRecyclableObject<VkBufferView> buff_view_;
    };

    VULKAN_DEFINE_IMP2(UnorderedAccessView, BufferUnorderedAccessView)

    class VulkanTextureUnorderedAccessView : public VulkanUnorderedAccessView
    {
    public:
        VulkanTextureUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format);
        VulkanTextureUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format);
        VulkanTextureUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format);

        ~VulkanTextureUnorderedAccessView() override;

        VulkanTextureUnorderedAccessView(VulkanTextureUnorderedAccessView&& other) noexcept;
        explicit VulkanTextureUnorderedAccessView(GpuUnorderedAccessViewInternal&& other) noexcept;
        explicit VulkanTextureUnorderedAccessView(VulkanUnorderedAccessView&& other) noexcept;
        VulkanTextureUnorderedAccessView& operator=(VulkanTextureUnorderedAccessView&& other) noexcept;
        GpuUnorderedAccessViewInternal& operator=(GpuUnorderedAccessViewInternal&& other) noexcept override;
        VulkanUnorderedAccessView& operator=(VulkanUnorderedAccessView&& other) noexcept override;

        bool IsBuffer() const noexcept override
        {
            return false;
        }

        void Reset() override;

        void Transition(VulkanCommandList& cmd_list) const override;

        VkImage Image() const noexcept;
        VkImageView ImageView() const noexcept;
        const VkImageSubresourceRange& SubresourceRange() const;

    private:
        void CreateImageView(VkDevice device, VkImage image, VkImageViewType type, GpuFormat format);
        void FillWriteDescSetForImage();

    private:
        uint32_t sub_resource_;

        VkDescriptorImageInfo image_info_{};
        VkImageSubresourceRange subres_range_{};
        VulkanRecyclableObject<VkImageView> image_view_;
    };

    VULKAN_DEFINE_IMP2(UnorderedAccessView, TextureUnorderedAccessView)

    VkWriteDescriptorSet NullSampledImageShaderResourceViewWriteDescSet();
    VkWriteDescriptorSet NullUniformTexelBufferShaderResourceViewWriteDescSet();

    VkWriteDescriptorSet NullStorageImageUnorderedAccessViewWriteDescSet();
    VkWriteDescriptorSet NullStorageTexelBufferUnorderedAccessViewWriteDescSet();
    VkWriteDescriptorSet NullStorageBufferUnorderedAccessViewWriteDescSet();

    VkWriteDescriptorSet NullUniformBufferConstantBufferViewWriteDescSet();
} // namespace AIHoloImager
