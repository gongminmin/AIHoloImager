// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <cstdint>

#include <directx/d3d12.h>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuDescriptorAllocator.hpp"
#include "Gpu/GpuFormat.hpp"

namespace AIHoloImager
{
    class GpuBuffer;
    class GpuCommandList;
    class GpuSystem;
    class GpuTexture2D;
    class GpuTexture2DArray;
    class GpuTexture3D;

    class GpuShaderResourceView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuShaderResourceView)

    public:
        GpuShaderResourceView() noexcept;
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, GpuFormat format);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, GpuFormat format);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, GpuFormat format);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource);
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format);
        // Typed buffer
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuBuffer& buffer, GpuFormat format);
        GpuShaderResourceView(
            GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format);
        // Structured buffer
        GpuShaderResourceView(GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t element_size);
        GpuShaderResourceView(
            GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);
        ~GpuShaderResourceView();

        GpuShaderResourceView(GpuShaderResourceView&& other) noexcept;
        GpuShaderResourceView& operator=(GpuShaderResourceView&& other) noexcept;

        void Reset();

        void Transition(GpuCommandList& cmd_list) const;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;
        const GpuTexture2D* texture_2d_ = nullptr;
        const GpuTexture2DArray* texture_2d_array_ = nullptr;
        const GpuTexture3D* texture_3d_ = nullptr;
        const GpuBuffer* buffer_ = nullptr;
        GpuDescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };

    class GpuRenderTargetView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuRenderTargetView)

    public:
        GpuRenderTargetView() noexcept;
        GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture);
        GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format);
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
        GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format);
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
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, GpuFormat format);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, GpuFormat format);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format);
        // Typed buffer
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, GpuFormat format);
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format);
        // Structured buffer
        GpuUnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t element_size);
        GpuUnorderedAccessView(
            GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);
        ~GpuUnorderedAccessView();

        GpuUnorderedAccessView(GpuUnorderedAccessView&& other) noexcept;
        GpuUnorderedAccessView& operator=(GpuUnorderedAccessView&& other) noexcept;

        void Reset();

        void Transition(GpuCommandList& cmd_list) const;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept;

        GpuTexture2D* Texture2D() const noexcept
        {
            return texture_2d_;
        }
        GpuTexture2DArray* Texture2DArray() const noexcept
        {
            return texture_2d_array_;
        }
        GpuTexture3D* Texture3D() const noexcept
        {
            return texture_3d_;
        }
        GpuBuffer* Buffer() const noexcept
        {
            return buffer_;
        }

    private:
        GpuSystem* gpu_system_ = nullptr;
        GpuTexture2D* texture_2d_ = nullptr;
        GpuTexture2DArray* texture_2d_array_ = nullptr;
        GpuTexture3D* texture_3d_ = nullptr;
        GpuBuffer* buffer_ = nullptr;
        GpuDescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };
} // namespace AIHoloImager
