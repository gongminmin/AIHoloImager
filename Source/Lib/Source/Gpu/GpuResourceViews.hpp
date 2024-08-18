// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>

#include <directx/d3d12.h>

#include "Gpu/GpuDescriptorAllocator.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuBuffer;
    class GpuCommandList;
    class GpuSystem;
    class GpuTexture2D;

    class GpuShaderResourceView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuShaderResourceView)

    public:
        GpuShaderResourceView() noexcept;
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, DXGI_FORMAT format);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, DXGI_FORMAT format);
        ~GpuShaderResourceView();

        GpuShaderResourceView(GpuShaderResourceView&& other) noexcept;
        GpuShaderResourceView& operator=(GpuShaderResourceView&& other) noexcept;

        void Reset();

        void Transition(GpuCommandList& cmd_list) const;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;
        const GpuTexture2D* texture_ = nullptr;
        GpuDescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    class GpuRenderTargetView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuRenderTargetView)

    public:
        GpuRenderTargetView() noexcept;
        GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture);
        GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture, DXGI_FORMAT format);
        ~GpuRenderTargetView();

        GpuRenderTargetView(GpuRenderTargetView&& other) noexcept;
        GpuRenderTargetView& operator=(GpuRenderTargetView&& other) noexcept;

        explicit operator bool() const noexcept;

        void Reset();

        void Transition(GpuCommandList& cmd_list) const;

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuTexture2D* texture_ = nullptr;
        GpuDescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    class GpuDepthStencilView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDepthStencilView)

    public:
        GpuDepthStencilView() noexcept;
        GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture);
        GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture, DXGI_FORMAT format);
        ~GpuDepthStencilView();

        GpuDepthStencilView(GpuDepthStencilView&& other) noexcept;
        GpuDepthStencilView& operator=(GpuDepthStencilView&& other) noexcept;

        explicit operator bool() const noexcept;

        void Reset();

        void Transition(GpuCommandList& cmd_list) const;

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuTexture2D* texture_ = nullptr;
        GpuDescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    class GpuUnorderedAccessView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuUnorderedAccessView)

    public:
        GpuUnorderedAccessView() noexcept;
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, DXGI_FORMAT format);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, DXGI_FORMAT format);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t element_size);
        GpuUnorderedAccessView(
            GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);
        ~GpuUnorderedAccessView();

        GpuUnorderedAccessView(GpuUnorderedAccessView&& other) noexcept;
        GpuUnorderedAccessView& operator=(GpuUnorderedAccessView&& other) noexcept;

        void Reset();

        void Transition(GpuCommandList& cmd_list) const;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuTexture2D* texture_ = nullptr;
        GpuBuffer* buffer_ = nullptr;
        GpuDescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };
} // namespace AIHoloImager
