// Copyright (c) 2026 Minmin Gong
//

#include "Gpu/GpuFence.hpp"

#include "Gpu/GpuSystem.hpp"

#include "Internal/GpuFenceInternal.hpp"
#include "Internal/GpuSystemInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    EMPTY_IMP(GpuFence)
    IMP_INTERNAL(GpuFence)

    GpuFence::GpuFence() noexcept = default;
    GpuFence::GpuFence(GpuSystem& gpu_system, uint64_t init_val, bool enable_sharing)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateFence(init_val, enable_sharing).release()))
    {
    }

    GpuFence::~GpuFence() noexcept = default;

    GpuFence::GpuFence(GpuFence&& other) noexcept = default;
    GpuFence& GpuFence::operator=(GpuFence&& other) noexcept = default;

    GpuFence::operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    void* GpuFence::NativeFence() const noexcept
    {
        return impl_->NativeFence();
    }

    void* GpuFence::SharedFenceHandle() const noexcept
    {
        return impl_->SharedFenceHandle();
    }

    uint64_t GpuFence::CompletedValue() const
    {
        return impl_->CompletedValue();
    }

    void GpuFence::CpuWait(uint64_t value) const
    {
        impl_->CpuWait(value);
    }
} // namespace AIHoloImager
