// Copyright (c) 2025 Minmin Gong
//

#include "Gpu/GpuCommandAllocatorInfo.hpp"

#include <cassert>

#include "Internal/GpuCommandAllocatorInfoInternal.hpp"
#include "Internal/GpuSystemInternal.hpp"

namespace AIHoloImager
{
    class GpuCommandAllocatorInfo::Impl : public GpuCommandAllocatorInfoInternal
    {
    };

    GpuCommandAllocatorInfo::GpuCommandAllocatorInfo() noexcept = default;

    GpuCommandAllocatorInfo::GpuCommandAllocatorInfo(GpuSystem& gpu_system, GpuSystem::CmdQueueType type)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateCommandAllocatorInfo(type).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuCommandAllocatorInfoInternal));
    }

    GpuCommandAllocatorInfo::~GpuCommandAllocatorInfo() noexcept = default;

    GpuCommandAllocatorInfo::GpuCommandAllocatorInfo(GpuCommandAllocatorInfo&& other) noexcept = default;
    GpuCommandAllocatorInfo& GpuCommandAllocatorInfo::operator=(GpuCommandAllocatorInfo&& other) noexcept = default;

    GpuCommandAllocatorInfoInternal& GpuCommandAllocatorInfo::Internal() noexcept
    {
        assert(impl_);
        return *impl_;
    }
} // namespace AIHoloImager
