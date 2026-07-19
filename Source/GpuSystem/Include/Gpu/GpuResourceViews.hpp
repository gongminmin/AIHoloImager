// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <cstdint>
#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuConstantBuffer.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResource.hpp"
#include "Gpu/InternalDefine.hpp"
#include "Gpu/Symbol.hpp"

namespace AIHoloImager
{
    class GpuBuffer;
    class GpuCommandList;
    class GpuSystem;
    class GpuTexture2D;
    class GpuTexture2DArray;
    class GpuTexture3D;

    class GpuConstantBufferViewInternal;

    class GpuConstantBufferView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuConstantBufferView)
        DEFINE_INTERNAL(GpuConstantBufferView)

    public:
        AIHI_GPU_SYS_API GpuConstantBufferView() noexcept;
        AIHI_GPU_SYS_API GpuConstantBufferView(GpuSystem& gpu_system, const GpuConstantBuffer& cbuffer);
        AIHI_GPU_SYS_API ~GpuConstantBufferView();

        AIHI_GPU_SYS_API GpuConstantBufferView(GpuConstantBufferView&& other) noexcept;
        AIHI_GPU_SYS_API GpuConstantBufferView& operator=(GpuConstantBufferView&& other) noexcept;

        AIHI_GPU_SYS_API void Reset();

        AIHI_GPU_SYS_API void Transition(GpuCommandList& cmd_list) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class GpuShaderResourceViewInternal;

    class GpuShaderResourceView final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuShaderResourceView)
        DEFINE_INTERNAL(GpuShaderResourceView)

    public:
        AIHI_GPU_SYS_API GpuShaderResourceView() noexcept;
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture);
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, GpuFormat format);
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource);
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format);
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array);
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, GpuFormat format);
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource);
        AIHI_GPU_SYS_API GpuShaderResourceView(
            GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format);
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture);
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, GpuFormat format);
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource);
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format);
        // Typed buffer
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuBuffer& buffer, GpuFormat format);
        AIHI_GPU_SYS_API GpuShaderResourceView(
            GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format);
        // Structured buffer
        AIHI_GPU_SYS_API GpuShaderResourceView(GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t element_size);
        AIHI_GPU_SYS_API GpuShaderResourceView(
            GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);
        AIHI_GPU_SYS_API ~GpuShaderResourceView();

        AIHI_GPU_SYS_API GpuShaderResourceView(GpuShaderResourceView&& other) noexcept;
        AIHI_GPU_SYS_API GpuShaderResourceView& operator=(GpuShaderResourceView&& other) noexcept;

        AIHI_GPU_SYS_API void Reset();

        AIHI_GPU_SYS_API void Transition(GpuCommandList& cmd_list) const;

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
        AIHI_GPU_SYS_API GpuRenderTargetView() noexcept;
        AIHI_GPU_SYS_API GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture);
        AIHI_GPU_SYS_API GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format);
        AIHI_GPU_SYS_API ~GpuRenderTargetView();

        AIHI_GPU_SYS_API GpuRenderTargetView(GpuRenderTargetView&& other) noexcept;
        AIHI_GPU_SYS_API GpuRenderTargetView& operator=(GpuRenderTargetView&& other) noexcept;

        AIHI_GPU_SYS_API explicit operator bool() const noexcept;

        AIHI_GPU_SYS_API void Reset();

        AIHI_GPU_SYS_API void Transition(GpuCommandList& cmd_list) const;

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
        AIHI_GPU_SYS_API GpuDepthStencilView() noexcept;
        AIHI_GPU_SYS_API GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture);
        AIHI_GPU_SYS_API GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format);
        AIHI_GPU_SYS_API ~GpuDepthStencilView();

        AIHI_GPU_SYS_API GpuDepthStencilView(GpuDepthStencilView&& other) noexcept;
        AIHI_GPU_SYS_API GpuDepthStencilView& operator=(GpuDepthStencilView&& other) noexcept;

        AIHI_GPU_SYS_API explicit operator bool() const noexcept;

        AIHI_GPU_SYS_API void Reset();

        AIHI_GPU_SYS_API void Transition(GpuCommandList& cmd_list) const;

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
        AIHI_GPU_SYS_API GpuUnorderedAccessView() noexcept;
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, GpuFormat format);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(
            GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, GpuFormat format);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format);
        // Typed buffer
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, GpuFormat format);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(
            GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format);
        // Structured buffer
        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t element_size);
        AIHI_GPU_SYS_API GpuUnorderedAccessView(
            GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size);
        AIHI_GPU_SYS_API ~GpuUnorderedAccessView();

        AIHI_GPU_SYS_API GpuUnorderedAccessView(GpuUnorderedAccessView&& other) noexcept;
        AIHI_GPU_SYS_API GpuUnorderedAccessView& operator=(GpuUnorderedAccessView&& other) noexcept;

        AIHI_GPU_SYS_API void Reset();

        AIHI_GPU_SYS_API void Transition(GpuCommandList& cmd_list) const;

        AIHI_GPU_SYS_API GpuResource* Resource() noexcept;
        AIHI_GPU_SYS_API const GpuResource* Resource() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
