// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>

#include <directx/d3d12.h>

#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuSystem;
    class GpuTexture2D;

    class GpuShaderResourceView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuShaderResourceView)

    public:
        GpuShaderResourceView() noexcept;
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle);
        GpuShaderResourceView(
            GpuSystem& gpu_system, const GpuTexture2D& texture, DXGI_FORMAT format, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle);
        GpuShaderResourceView(
            GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, DXGI_FORMAT format,
            D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle);
        ~GpuShaderResourceView() noexcept;

        GpuShaderResourceView(GpuShaderResourceView&& other) noexcept;
        GpuShaderResourceView& operator=(GpuShaderResourceView&& other) noexcept;

        explicit operator bool() const noexcept;

        void Reset() noexcept;

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

    private:
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    class GpuRenderTargetView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuRenderTargetView)

    public:
        GpuRenderTargetView() noexcept;
        GpuRenderTargetView(GpuSystem& gpu_system, const GpuTexture2D& texture, DXGI_FORMAT format, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle);
        ~GpuRenderTargetView() noexcept;

        GpuRenderTargetView(GpuRenderTargetView&& other) noexcept;
        GpuRenderTargetView& operator=(GpuRenderTargetView&& other) noexcept;

        explicit operator bool() const noexcept;

        void Reset() noexcept;

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

    private:
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    class GpuDepthStencilView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDepthStencilView)

    public:
        GpuDepthStencilView() noexcept;
        GpuDepthStencilView(GpuSystem& gpu_system, const GpuTexture2D& texture, DXGI_FORMAT format, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle);
        ~GpuDepthStencilView() noexcept;

        GpuDepthStencilView(GpuDepthStencilView&& other) noexcept;
        GpuDepthStencilView& operator=(GpuDepthStencilView&& other) noexcept;

        explicit operator bool() const noexcept;

        void Reset() noexcept;

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

    private:
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    class GpuUnorderedAccessView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuUnorderedAccessView)

    public:
        GpuUnorderedAccessView() noexcept;
        GpuUnorderedAccessView(GpuSystem& gpu_system, const GpuTexture2D& texture, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle);
        GpuUnorderedAccessView(
            GpuSystem& gpu_system, const GpuTexture2D& texture, DXGI_FORMAT format, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle);
        GpuUnorderedAccessView(
            GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle);
        GpuUnorderedAccessView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, DXGI_FORMAT format,
            D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle);
        ~GpuUnorderedAccessView() noexcept;

        GpuUnorderedAccessView(GpuUnorderedAccessView&& other) noexcept;
        GpuUnorderedAccessView& operator=(GpuUnorderedAccessView&& other) noexcept;

        explicit operator bool() const noexcept;

        void Reset() noexcept;

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

    private:
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };
} // namespace AIHoloImager
