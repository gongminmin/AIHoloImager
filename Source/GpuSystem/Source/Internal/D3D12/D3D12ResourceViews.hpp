// Copyright (c) 2025-2026 Minmin Gong
//

#pragma once

#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"

#include "../GpuResourceViewsInternal.hpp"
#include "D3D12CommandList.hpp"
#include "D3D12DescriptorAllocator.hpp"

namespace AIHoloImager
{
    class D3D12ConstantBufferView : public GpuConstantBufferViewInternal
    {
    public:
        D3D12ConstantBufferView(const GpuBuffer& buffer, uint32_t offset, uint32_t size);

        ~D3D12ConstantBufferView() override;

        D3D12ConstantBufferView(D3D12ConstantBufferView&& other) noexcept;
        explicit D3D12ConstantBufferView(GpuConstantBufferViewInternal&& other) noexcept;
        D3D12ConstantBufferView& operator=(D3D12ConstantBufferView&& other) noexcept;
        GpuConstantBufferViewInternal& operator=(GpuConstantBufferViewInternal&& other) noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list) const override;
        void Transition(D3D12CommandList& cmd_list) const;

        D3D12_GPU_VIRTUAL_ADDRESS GpuVirtualAddress() const noexcept;

    private:
        const GpuResource* resource_ = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS gpu_virtual_addr_;
    };

    D3D12_DEFINE_IMP(ConstantBufferView)

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
        void Transition(D3D12CommandList& cmd_list) const;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;
        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

        const GpuResource* Resource() const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;
        const GpuResource* resource_ = nullptr;
        uint32_t sub_resource_;
        D3D12DescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    D3D12_DEFINE_IMP(ShaderResourceView)

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
        void Transition(D3D12CommandList& cmd_list) const;
        void TransitionBack(D3D12CommandList& cmd_list) const;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;
        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuResource* resource_ = nullptr;
        uint32_t sub_resource_;
        D3D12DescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    D3D12_DEFINE_IMP(RenderTargetView)

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
        void Transition(D3D12CommandList& cmd_list) const;
        void TransitionBack(D3D12CommandList& cmd_list) const;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;
        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuResource* resource_ = nullptr;
        uint32_t sub_resource_;
        D3D12DescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    D3D12_DEFINE_IMP(DepthStencilView)

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
        void Transition(D3D12CommandList& cmd_list) const;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;
        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

        GpuResource* Resource() noexcept override;

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuResource* resource_ = nullptr;
        uint32_t sub_resource_;
        D3D12DescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    D3D12_DEFINE_IMP(UnorderedAccessView)
} // namespace AIHoloImager
