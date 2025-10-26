// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <cstdint>
#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuDescriptorHeap.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResource.hpp"
#include "Gpu/InternalDefine.hpp"

namespace AIHoloImager
{
    class GpuBuffer;
    class GpuCommandList;
    class GpuSystem;
    class GpuTexture2D;
    class GpuTexture2DArray;
    class GpuTexture3D;

    class GpuShaderResourceViewInternal;

    class GpuShaderResourceView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuShaderResourceView)
        DEFINE_INTERNAL(GpuShaderResourceView)

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

        void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept;
        GpuDescriptorCpuHandle CpuHandle() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class GpuRenderTargetViewInternal;

    class GpuRenderTargetView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuRenderTargetView)
        DEFINE_INTERNAL(GpuRenderTargetView)

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

        void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept;
        GpuDescriptorCpuHandle CpuHandle() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class GpuDepthStencilViewInternal;

    class GpuDepthStencilView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDepthStencilView)
        DEFINE_INTERNAL(GpuDepthStencilView)

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

        void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept;
        GpuDescriptorCpuHandle CpuHandle() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class GpuUnorderedAccessViewInternal;

    class GpuUnorderedAccessView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuUnorderedAccessView)
        DEFINE_INTERNAL(GpuUnorderedAccessView)

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

        void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept;
        GpuDescriptorCpuHandle CpuHandle() const noexcept;

        GpuResource* Resource() noexcept;
        const GpuResource* Resource() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
