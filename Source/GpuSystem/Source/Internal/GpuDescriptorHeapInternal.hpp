// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuDescriptorHeap.hpp"

namespace AIHoloImager
{
    class GpuDescriptorHeapInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDescriptorHeapInternal)

    public:
        GpuDescriptorHeapInternal() noexcept;
        virtual ~GpuDescriptorHeapInternal();

        GpuDescriptorHeapInternal(GpuDescriptorHeapInternal&& other) noexcept;
        virtual GpuDescriptorHeapInternal& operator=(GpuDescriptorHeapInternal&& other) noexcept = 0;

        virtual void Name(std::string_view name) = 0;

        virtual void* NativeDescriptorHeap() const noexcept = 0;

        virtual GpuDescriptorHeapType Type() const noexcept = 0;

        virtual GpuDescriptorCpuHandle CpuHandleStart() const noexcept = 0;
        virtual GpuDescriptorGpuHandle GpuHandleStart() const noexcept = 0;

        virtual uint32_t Size() const noexcept = 0;

        virtual void Reset() noexcept = 0;
    };
} // namespace AIHoloImager
