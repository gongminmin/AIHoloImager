// Copyright (c) 2024-2005 Minmin Gong
//

#pragma once

#include <cstdint>

#include "Base/Enum.hpp"

namespace AIHoloImager
{
    class GpuCommandList;

    enum class GpuHeap
    {
        Default,
        Upload,
        ReadBack,
    };

    enum class GpuResourceFlag : uint32_t
    {
        None = 0,
        RenderTarget = 1U << 0,
        DepthStencil = 1U << 1,
        UnorderedAccess = 1U << 2,
        Shareable = 1U << 3,
    };
    ENUM_CLASS_BITWISE_OPERATORS(GpuResourceFlag);

    enum class GpuResourceState
    {
        Common,

        ColorWrite,
        DepthWrite,

        UnorderedAccess,

        CopySrc,
        CopyDst,

        RayTracingAS,
    };

    enum class GpuResourceType
    {
        Buffer,
        Texture2D,
        Texture2DArray,
        Texture3D,
    };

    class GpuSystem;
    class GpuResourceInternal;

    class GpuResource
    {
    public:
        virtual ~GpuResource() noexcept;

        virtual void Name(std::wstring_view name) = 0;

        virtual void* NativeResource() const noexcept = 0;
        template <typename Traits>
        typename Traits::ResourceType NativeResource() const noexcept
        {
            return reinterpret_cast<typename Traits::ResourceType>(this->NativeResource());
        }

        virtual explicit operator bool() const noexcept = 0;

        virtual void Reset() = 0;

        virtual void* SharedHandle() const noexcept = 0;
        template <typename Traits>
        typename Traits::SharedHandleType SharedHandle() const noexcept
        {
            return reinterpret_cast<typename Traits::SharedHandleType>(this->SharedHandle());
        }

        virtual GpuResourceType Type() const noexcept = 0;

        virtual void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const = 0;
        virtual void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const = 0;

        virtual GpuResourceInternal& Internal() noexcept = 0;
        virtual const GpuResourceInternal& Internal() const noexcept = 0;
    };
} // namespace AIHoloImager
