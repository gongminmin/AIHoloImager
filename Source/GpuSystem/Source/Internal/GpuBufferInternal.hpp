// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuCommandList.hpp"

namespace AIHoloImager
{
    class GpuBufferInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuBufferInternal)

    public:
        GpuBufferInternal() noexcept;
        virtual ~GpuBufferInternal();

        GpuBufferInternal(GpuBufferInternal&& other) noexcept;
        virtual GpuBufferInternal& operator=(GpuBufferInternal&& other) noexcept = 0;

        virtual void Name(std::wstring_view name) = 0;

        virtual void* NativeResource() const noexcept = 0;
        virtual void* NativeBuffer() const noexcept = 0;

        virtual void* SharedHandle() const noexcept = 0;

        virtual GpuVirtualAddressType GpuVirtualAddress() const noexcept = 0;
        virtual uint32_t Size() const noexcept = 0;

        virtual void* Map(const GpuRange& read_range) = 0;
        virtual void* Map() = 0;
        virtual void Unmap(const GpuRange& write_range) = 0;
        virtual void Unmap() = 0;

        virtual void Reset() = 0;

        virtual void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const = 0;
    };
} // namespace AIHoloImager
