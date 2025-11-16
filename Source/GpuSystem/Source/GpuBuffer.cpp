// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuBuffer.hpp"

#include <cassert>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"

#include "Internal/GpuBufferInternal.hpp"
#include "Internal/GpuSystemInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    EMPTY_IMP(GpuBuffer)
    IMP_INTERNAL2(GpuBuffer, GpuResource)

    GpuBuffer::GpuBuffer() noexcept = default;

    GpuBuffer::GpuBuffer(GpuSystem& gpu_system, uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateBuffer(size, heap, flags, std::move(name)).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuBufferInternal));
    }

    GpuBuffer::GpuBuffer(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::string_view name)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateBuffer(native_resource, curr_state, std::move(name)).release()))
    {
    }

    GpuBuffer::~GpuBuffer() = default;

    GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept = default;
    GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept = default;

    void GpuBuffer::Name(std::string_view name)
    {
        assert(impl_);
        return impl_->Name(std::move(name));
    }

    void* GpuBuffer::NativeResource() const noexcept
    {
        return impl_ ? impl_->NativeResource() : nullptr;
    }

    void* GpuBuffer::NativeBuffer() const noexcept
    {
        return impl_ ? impl_->NativeBuffer() : nullptr;
    }

    GpuBuffer::operator bool() const noexcept
    {
        return this->NativeBuffer() != nullptr;
    }

    void* GpuBuffer::SharedHandle() const noexcept
    {
        return impl_ ? impl_->SharedHandle() : nullptr;
    }

    GpuHeap GpuBuffer::Heap() const noexcept
    {
        assert(impl_);
        return impl_->Heap();
    }

    GpuResourceType GpuBuffer::Type() const noexcept
    {
        assert(impl_);
        return impl_->Type();
    }

    uint32_t GpuBuffer::AllocationSize() const noexcept
    {
        assert(impl_);
        return impl_->AllocationSize();
    }

    uint32_t GpuBuffer::Size() const noexcept
    {
        return impl_ ? impl_->Size() : 0;
    }

    void* GpuBuffer::Map(const GpuRange& read_range)
    {
        assert(impl_);
        return impl_->Map(read_range);
    }

    const void* GpuBuffer::Map(const GpuRange& read_range) const
    {
        return const_cast<GpuBuffer*>(this)->Map(read_range);
    }

    void* GpuBuffer::Map()
    {
        assert(impl_);
        return impl_->Map();
    }

    const void* GpuBuffer::Map() const
    {
        return const_cast<GpuBuffer*>(this)->Map();
    }

    void GpuBuffer::Unmap(const GpuRange& write_range)
    {
        assert(impl_);
        impl_->Unmap(write_range);
    }

    void GpuBuffer::Unmap()
    {
        this->Unmap(GpuRange{0, 0});
    }

    void GpuBuffer::Unmap() const
    {
        return const_cast<GpuBuffer*>(this)->Unmap();
    }

    void GpuBuffer::Reset()
    {
        assert(impl_);
        impl_->Reset();
    }

    void GpuBuffer::Transition(GpuCommandList& cmd_list, [[maybe_unused]] uint32_t sub_resource, GpuResourceState target_state) const
    {
        assert(sub_resource == 0);
        this->Transition(cmd_list, target_state);
    }

    void GpuBuffer::Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const
    {
        assert(impl_);
        impl_->Transition(cmd_list, target_state);
    }
} // namespace AIHoloImager
