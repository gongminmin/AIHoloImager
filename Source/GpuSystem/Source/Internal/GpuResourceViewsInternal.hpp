// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuCommandList.hpp"

namespace AIHoloImager
{
    class GpuShaderResourceViewInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuShaderResourceViewInternal)

    public:
        GpuShaderResourceViewInternal() noexcept;
        virtual ~GpuShaderResourceViewInternal() noexcept;

        GpuShaderResourceViewInternal(GpuShaderResourceViewInternal&& other) noexcept;
        virtual GpuShaderResourceViewInternal& operator=(GpuShaderResourceViewInternal&& other) noexcept = 0;

        virtual void Reset() = 0;

        virtual void Transition(GpuCommandList& cmd_list) const = 0;

        virtual void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept = 0;
        virtual GpuDescriptorCpuHandle CpuHandle() const noexcept = 0;
    };

    class GpuRenderTargetViewInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuRenderTargetViewInternal)

    public:
        GpuRenderTargetViewInternal() noexcept;
        virtual ~GpuRenderTargetViewInternal() noexcept;

        GpuRenderTargetViewInternal(GpuRenderTargetViewInternal&& other) noexcept;
        virtual GpuRenderTargetViewInternal& operator=(GpuRenderTargetViewInternal&& other) noexcept = 0;

        virtual explicit operator bool() const noexcept = 0;

        virtual void Reset() = 0;

        virtual void Transition(GpuCommandList& cmd_list) const = 0;

        virtual void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept = 0;
        virtual GpuDescriptorCpuHandle CpuHandle() const noexcept = 0;
    };

    class GpuDepthStencilViewInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDepthStencilViewInternal)

    public:
        GpuDepthStencilViewInternal() noexcept;
        virtual ~GpuDepthStencilViewInternal() noexcept;

        GpuDepthStencilViewInternal(GpuDepthStencilViewInternal&& other) noexcept;
        virtual GpuDepthStencilViewInternal& operator=(GpuDepthStencilViewInternal&& other) noexcept = 0;

        virtual explicit operator bool() const noexcept = 0;

        virtual void Reset() = 0;

        virtual void Transition(GpuCommandList& cmd_list) const = 0;

        virtual void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept = 0;
        virtual GpuDescriptorCpuHandle CpuHandle() const noexcept = 0;
    };

    class GpuUnorderedAccessViewInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuUnorderedAccessViewInternal)

    public:
        GpuUnorderedAccessViewInternal() noexcept;
        virtual ~GpuUnorderedAccessViewInternal() noexcept;

        GpuUnorderedAccessViewInternal(GpuUnorderedAccessViewInternal&& other) noexcept;
        virtual GpuUnorderedAccessViewInternal& operator=(GpuUnorderedAccessViewInternal&& other) noexcept = 0;

        virtual void Reset() = 0;

        virtual void Transition(GpuCommandList& cmd_list) const = 0;

        virtual void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept = 0;
        virtual GpuDescriptorCpuHandle CpuHandle() const noexcept = 0;

        virtual GpuResource* Resource() noexcept = 0;
    };
} // namespace AIHoloImager
