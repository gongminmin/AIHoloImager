// Copyright (c) 2025 Minmin Gong
//

#include "Gpu/GpuCommandPool.hpp"

#include "Internal/GpuCommandPoolInternal.hpp"
#include "Internal/GpuSystemInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    EMPTY_IMP(GpuCommandPool)
    IMP_INTERNAL(GpuCommandPool)

    GpuCommandPool::GpuCommandPool() noexcept = default;

    GpuCommandPool::GpuCommandPool(GpuSystem& gpu_system, GpuSystem::CmdQueueType type)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateCommandPool(type).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuCommandPoolInternal));
    }

    GpuCommandPool::~GpuCommandPool() noexcept = default;

    GpuCommandPool::GpuCommandPool(GpuCommandPool&& other) noexcept = default;
    GpuCommandPool& GpuCommandPool::operator=(GpuCommandPool&& other) noexcept = default;
} // namespace AIHoloImager
