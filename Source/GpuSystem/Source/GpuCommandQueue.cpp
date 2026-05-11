// Copyright (c) 2026 Minmin Gong
//

#include "Gpu/GpuCommandQueue.hpp"

#include "Internal/GpuCommandQueueInternal.hpp"
#include "Internal/GpuSystemInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    EMPTY_IMP(GpuCommandQueue)
    IMP_INTERNAL(GpuCommandQueue)

    GpuCommandQueue::GpuCommandQueue() noexcept = default;

    GpuCommandQueue::GpuCommandQueue(GpuSystem& gpu_system, GpuSystem::CmdQueueType type, std::string_view name)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateCommandQueue(type, std::move(name)).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuCommandQueueInternal));
    }

    GpuCommandQueue::~GpuCommandQueue() noexcept = default;

    GpuCommandQueue::GpuCommandQueue(GpuCommandQueue&& other) noexcept = default;
    GpuCommandQueue& GpuCommandQueue::operator=(GpuCommandQueue&& other) noexcept = default;

    GpuCommandQueue::operator bool() const noexcept
    {
        return impl_ && static_cast<bool>(*impl_);
    }

    void* GpuCommandQueue::NativeCmdQueue() const noexcept
    {
        return impl_ ? impl_->NativeCmdQueue() : nullptr;
    }

    void GpuCommandQueue::GpuWait(std::span<const FenceInfo> fences)
    {
        impl_->GpuWait(fences);
    }

    void GpuCommandQueue::Execute(const GpuCommandList& cmd_list, std::span<const FenceInfo> wait_fences, const FenceInfo& signal_fence)
    {
        impl_->Execute(cmd_list, std::move(wait_fences), signal_fence);
    }
} // namespace AIHoloImager
