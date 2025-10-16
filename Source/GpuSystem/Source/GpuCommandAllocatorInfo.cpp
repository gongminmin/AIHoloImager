// Copyright (c) 2025 Minmin Gong
//

#include "Gpu/GpuCommandAllocatorInfo.hpp"

#include <cassert>

#include "Internal/GpuSystemInternalFactory.hpp"

namespace AIHoloImager
{
    class GpuCommandAllocatorInfo::Impl : public GpuCommandAllocatorInfoInternal
    {
    };

    GpuCommandAllocatorInfo::GpuCommandAllocatorInfo(GpuSystem& gpu_system)
        : impl_(static_cast<Impl*>(gpu_system.InternalFactory().CreateCommandAllocatorInfo().release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuCommandAllocatorInfoInternal));
    }

    GpuCommandAllocatorInfo::~GpuCommandAllocatorInfo() noexcept = default;

    GpuCommandAllocatorInfoInternal& GpuCommandAllocatorInfo::Internal() noexcept
    {
        assert(impl_);
        return *impl_;
    }
} // namespace AIHoloImager
