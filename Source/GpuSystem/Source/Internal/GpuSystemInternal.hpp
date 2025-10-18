// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuSystem.hpp"

#include "GpuCommandListInternal.hpp"

namespace AIHoloImager
{
    class GpuSystemInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuSystemInternal)

    public:
        GpuSystemInternal() noexcept;
        virtual ~GpuSystemInternal();

        GpuSystemInternal(GpuSystemInternal&& other) noexcept;
        virtual GpuSystemInternal& operator=(GpuSystemInternal&& other) noexcept = 0;

        virtual void* NativeDevice() const noexcept = 0;
        virtual void* NativeCommandQueue(GpuSystem::CmdQueueType type) const noexcept = 0;

        virtual void* SharedFenceHandle() const noexcept = 0;

        virtual [[nodiscard]] GpuCommandList CreateCommandList(GpuSystem::CmdQueueType type) = 0;
        virtual uint64_t Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value) = 0;
        virtual uint64_t ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value) = 0;
        virtual uint64_t ExecuteAndReset(GpuCommandListInternal& cmd_list, uint64_t wait_fence_value) = 0;

        virtual uint32_t ConstantDataAlignment() const noexcept = 0;
        virtual uint32_t StructuredDataAlignment() const noexcept = 0;
        virtual uint32_t TextureDataAlignment() const noexcept = 0;

        virtual void CpuWait(uint64_t fence_value) = 0;
        virtual void GpuWait(GpuSystem::CmdQueueType type, uint64_t fence_value) = 0;
        virtual uint64_t FenceValue() const noexcept = 0;
        virtual uint64_t CompletedFenceValue() const = 0;

        virtual void HandleDeviceLost() = 0;
        virtual void ClearStallResources() = 0;
    };
} // namespace AIHoloImager
