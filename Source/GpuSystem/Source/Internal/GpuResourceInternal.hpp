// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResource.hpp"

#include "GpuCommandListInternal.hpp"

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

        virtual void* SharedHandle() const noexcept = 0;

        virtual void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const = 0;
        virtual void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const = 0;
        virtual void Transition(GpuCommandListInternal& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const = 0;
        virtual void Transition(GpuCommandListInternal& cmd_list, GpuResourceState target_state) const = 0;
    };
} // namespace AIHoloImager
