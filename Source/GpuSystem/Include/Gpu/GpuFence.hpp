// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/InternalDefine.hpp"
#include "Gpu/Symbol.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class GpuFenceInternal;

    class GpuFence final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuFence)
        DEFINE_INTERNAL(GpuFence)

    public:
        AIHI_GPU_SYS_API GpuFence() noexcept;
        AIHI_GPU_SYS_API explicit GpuFence(GpuSystem& gpu_system, uint64_t init_val, bool enable_sharing);
        AIHI_GPU_SYS_API ~GpuFence() noexcept;

        AIHI_GPU_SYS_API GpuFence(GpuFence&& other) noexcept;
        AIHI_GPU_SYS_API GpuFence& operator=(GpuFence&& other) noexcept;

        AIHI_GPU_SYS_API explicit operator bool() const noexcept;

        AIHI_GPU_SYS_API void* NativeFence() const noexcept;
        template <typename Traits>
        typename Traits::FenceType NativeFence() const noexcept
        {
            return reinterpret_cast<typename Traits::FenceType>(this->NativeFence());
        }

        AIHI_GPU_SYS_API void* SharedFenceHandle() const noexcept;
        template <typename Traits>
        typename Traits::SharedHandleType SharedFenceHandle() const noexcept
        {
            return reinterpret_cast<typename Traits::SharedHandleType>(this->SharedFenceHandle());
        }

        AIHI_GPU_SYS_API uint64_t CompletedValue() const;

        AIHI_GPU_SYS_API void CpuWait(uint64_t value) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
