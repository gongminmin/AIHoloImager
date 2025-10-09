// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuResource.hpp"

namespace AIHoloImager
{
    class GpuResourceInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuResourceInternal)

    public:
        GpuResourceInternal() noexcept;
        virtual ~GpuResourceInternal();

        GpuResourceInternal(GpuResourceInternal&& other) noexcept;

        virtual GpuResourceInternal& operator=(GpuResourceInternal&& other) noexcept = 0;

        virtual void Name(std::wstring_view name) = 0;

        virtual void* NativeResource() const noexcept = 0;

        virtual void Reset() = 0;

        virtual void CreateResource(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size,
            uint32_t mip_levels, GpuFormat format, GpuHeap heap, GpuResourceFlag flags, GpuResourceState init_state,
            std::wstring_view name) = 0;

        virtual void* SharedHandle() const noexcept = 0;

        virtual GpuResourceType Type() const noexcept = 0;

        virtual uint32_t Width() const noexcept = 0;
        virtual uint32_t Height() const noexcept = 0;
        virtual uint32_t Depth() const noexcept = 0;
        virtual uint32_t ArraySize() const noexcept = 0;
        virtual uint32_t MipLevels() const noexcept = 0;

        virtual GpuFormat Format() const noexcept = 0;

        virtual GpuResourceFlag Flags() const noexcept = 0;
    };
} // namespace AIHoloImager
