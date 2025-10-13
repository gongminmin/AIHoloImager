// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"

#include "../GpuResourceViewsInternal.hpp"

namespace AIHoloImager
{
    class D3D12ShaderResourceView : public GpuShaderResourceViewInternal
    {
    public:
        D3D12ShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format);
        D3D12ShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format);
        D3D12ShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format);

        D3D12ShaderResourceView(
            GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format);
        D3D12ShaderResourceView(
            GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);

        ~D3D12ShaderResourceView() override;

        D3D12ShaderResourceView(D3D12ShaderResourceView&& other) noexcept;
        explicit D3D12ShaderResourceView(GpuShaderResourceViewInternal&& other) noexcept;
        D3D12ShaderResourceView& operator=(D3D12ShaderResourceView&& other) noexcept;
        GpuShaderResourceViewInternal& operator=(GpuShaderResourceViewInternal&& other) noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list) const override;

        void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept override;
        GpuDescriptorCpuHandle CpuHandle() const noexcept override;

    private:
        GpuSystem* gpu_system_ = nullptr;
        const GpuResource* resource_ = nullptr;
        GpuDescriptorBlock desc_block_;
        GpuDescriptorCpuHandle cpu_handle_{};
    };

    class D3D12RenderTargetView : public GpuRenderTargetViewInternal
    {
    public:
        D3D12RenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format);
        ~D3D12RenderTargetView() override;

        D3D12RenderTargetView(D3D12RenderTargetView&& other) noexcept;
        explicit D3D12RenderTargetView(GpuRenderTargetViewInternal&& other) noexcept;
        D3D12RenderTargetView& operator=(D3D12RenderTargetView&& other) noexcept;
        GpuRenderTargetViewInternal& operator=(GpuRenderTargetViewInternal&& other) noexcept override;

        explicit operator bool() const noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list) const override;

        void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept override;
        GpuDescriptorCpuHandle CpuHandle() const noexcept override;

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuResource* resource_ = nullptr;
        GpuDescriptorBlock desc_block_;
        GpuDescriptorCpuHandle cpu_handle_{};
    };

    class D3D12DepthStencilView : public GpuDepthStencilViewInternal
    {
    public:
        D3D12DepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format);
        ~D3D12DepthStencilView() override;

        D3D12DepthStencilView(D3D12DepthStencilView&& other) noexcept;
        explicit D3D12DepthStencilView(GpuDepthStencilViewInternal&& other) noexcept;
        D3D12DepthStencilView& operator=(D3D12DepthStencilView&& other) noexcept;
        GpuDepthStencilViewInternal& operator=(GpuDepthStencilViewInternal&& other) noexcept override;

        explicit operator bool() const noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list) const override;

        void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept override;
        GpuDescriptorCpuHandle CpuHandle() const noexcept override;

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuResource* resource_ = nullptr;
        GpuDescriptorBlock desc_block_;
        GpuDescriptorCpuHandle cpu_handle_{};
    };

    class D3D12UnorderedAccessView : public GpuUnorderedAccessViewInternal
    {
    public:
        D3D12UnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format);
        D3D12UnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format);
        D3D12UnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format);

        D3D12UnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format);

        D3D12UnorderedAccessView(
            GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);

        ~D3D12UnorderedAccessView() override;

        D3D12UnorderedAccessView(D3D12UnorderedAccessView&& other) noexcept;
        explicit D3D12UnorderedAccessView(GpuUnorderedAccessViewInternal&& other) noexcept;
        D3D12UnorderedAccessView& operator=(D3D12UnorderedAccessView&& other) noexcept;
        GpuUnorderedAccessViewInternal& operator=(GpuUnorderedAccessViewInternal&& other) noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list) const override;

        void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept override;
        GpuDescriptorCpuHandle CpuHandle() const noexcept override;

        GpuResource* Resource() noexcept override;

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuResource* resource_ = nullptr;
        GpuDescriptorBlock desc_block_;
        GpuDescriptorCpuHandle cpu_handle_{};
    };
} // namespace AIHoloImager
