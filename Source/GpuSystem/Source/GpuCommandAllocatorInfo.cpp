// Copyright (c) 2025 Minmin Gong
//

#include "Gpu/GpuCommandAllocatorInfo.hpp"

#include "Internal/GpuCommandAllocatorInfoInternal.hpp"
#include "Internal/GpuSystemInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    EMPTY_IMP(GpuCommandAllocatorInfo)
    IMP_INTERNAL(GpuCommandAllocatorInfo)

    GpuCommandAllocatorInfo::GpuCommandAllocatorInfo() noexcept = default;

    GpuCommandAllocatorInfo::GpuCommandAllocatorInfo(GpuSystem& gpu_system, GpuSystem::CmdQueueType type)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateCommandAllocatorInfo(type).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuCommandAllocatorInfoInternal));
    }

    GpuCommandAllocatorInfo::~GpuCommandAllocatorInfo() noexcept = default;

    GpuCommandAllocatorInfo::GpuCommandAllocatorInfo(GpuCommandAllocatorInfo&& other) noexcept = default;
    GpuCommandAllocatorInfo& GpuCommandAllocatorInfo::operator=(GpuCommandAllocatorInfo&& other) noexcept = default;
} // namespace AIHoloImager
