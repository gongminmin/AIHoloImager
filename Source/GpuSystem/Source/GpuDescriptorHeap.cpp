// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuDescriptorHeap.hpp"

#include <cassert>

#include "Gpu/GpuSystem.hpp"

#include "Internal/GpuDescriptorHeapInternal.hpp"
#include "Internal/GpuSystemInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    GpuDescriptorCpuHandle OffsetHandle(const GpuDescriptorCpuHandle& handle, int32_t offset, uint32_t desc_size)
    {
        return {handle.handle + offset * desc_size};
    }

    GpuDescriptorGpuHandle OffsetHandle(const GpuDescriptorGpuHandle& handle, int32_t offset, uint32_t desc_size)
    {
        return {handle.handle + offset * desc_size};
    }

    std::tuple<GpuDescriptorCpuHandle, GpuDescriptorGpuHandle> OffsetHandle(
        const GpuDescriptorCpuHandle& cpu_handle, const GpuDescriptorGpuHandle& gpu_handle, int32_t offset, uint32_t desc_size)
    {
        const int32_t offset_in_bytes = offset * desc_size;
        return {{cpu_handle.handle + offset_in_bytes}, {gpu_handle.handle + offset_in_bytes}};
    }


    EMPTY_IMP(GpuDescriptorHeap)
    IMP_INTERNAL(GpuDescriptorHeap)

    GpuDescriptorHeap::GpuDescriptorHeap() noexcept = default;
    GpuDescriptorHeap::GpuDescriptorHeap(
        GpuSystem& gpu_system, uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateDescriptorHeap(size, type, shader_visible, std::move(name)).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuDescriptorHeapInternal));
    }

    GpuDescriptorHeap::~GpuDescriptorHeap() noexcept = default;

    GpuDescriptorHeap::GpuDescriptorHeap(GpuDescriptorHeap&& other) noexcept = default;
    GpuDescriptorHeap& GpuDescriptorHeap::operator=(GpuDescriptorHeap&& other) noexcept = default;

    void GpuDescriptorHeap::Name(std::wstring_view name)
    {
        assert(impl_);
        impl_->Name(std::move(name));
    }

    GpuDescriptorHeap::operator bool() const noexcept
    {
        return this->NativeDescriptorHeap() != nullptr;
    }

    void* GpuDescriptorHeap::NativeDescriptorHeap() const noexcept
    {
        return impl_ ? impl_->NativeDescriptorHeap() : nullptr;
    }

    GpuDescriptorHeapType GpuDescriptorHeap::Type() const noexcept
    {
        assert(impl_);
        return impl_->Type();
    }

    GpuDescriptorCpuHandle GpuDescriptorHeap::CpuHandleStart() const noexcept
    {
        assert(impl_);
        return impl_->CpuHandleStart();
    }

    GpuDescriptorGpuHandle GpuDescriptorHeap::GpuHandleStart() const noexcept
    {
        assert(impl_);
        return impl_->GpuHandleStart();
    }

    uint32_t GpuDescriptorHeap::Size() const noexcept
    {
        return impl_ ? impl_->Size() : 0;
    }

    void GpuDescriptorHeap::Reset() noexcept
    {
        assert(impl_);
        impl_->Reset();
    }
} // namespace AIHoloImager
