// Copyright (c) 2024-2005 Minmin Gong
//

#pragma once

#include <cstdint>

#include "Base/Enum.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuFormat.hpp"

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

    class GpuResource
    {
        DISALLOW_COPY_AND_ASSIGN(GpuResource)

    public:
        GpuResource() noexcept;
        explicit GpuResource(GpuSystem& gpu_system);
        GpuResource(GpuSystem& gpu_system, void* native_resource, std::wstring_view name = L"");
        template <typename Traits>
        GpuResource(GpuSystem& gpu_system, typename Traits::ResourceType native_resource, std::wstring_view name = L"")
            : GpuResource(gpu_system, reinterpret_cast<void*>(native_resource), std::move(name))
        {
        }
        virtual ~GpuResource();

        GpuResource(GpuResource&& other) noexcept;
        GpuResource& operator=(GpuResource&& other) noexcept;

        void Name(std::wstring_view name);

        void* NativeResource() const noexcept;
        template <typename Traits>
        typename Traits::ResourceType NativeResource() const noexcept
        {
            return reinterpret_cast<typename Traits::ResourceType>(this->NativeResource());
        }

        explicit operator bool() const noexcept;

        void Reset();

        void* SharedHandle() const noexcept;
        template <typename Traits>
        typename Traits::SharedHandleType SharedHandle() const noexcept
        {
            return reinterpret_cast<typename Traits::SharedHandleType>(this->SharedHandle());
        }

        virtual void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const = 0;
        virtual void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const = 0;

    protected:
        void CreateResource(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size, uint32_t mip_levels,
            GpuFormat format, GpuHeap heap, GpuResourceFlag flags, GpuResourceState init_state, std::wstring_view name);

        GpuResourceType Type() const noexcept;
        uint32_t Width() const noexcept;
        uint32_t Height() const noexcept;
        uint32_t Depth() const noexcept;
        uint32_t ArraySize() const noexcept;
        uint32_t MipLevels() const noexcept;
        GpuFormat Format() const noexcept;
        GpuResourceFlag Flags() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
