// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/InternalDefine.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class GpuFenceInternal;

    class GpuFence final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuFence)
        DEFINE_INTERNAL(GpuFence)

    public:
        GpuFence() noexcept;
        explicit GpuFence(GpuSystem& gpu_system, uint64_t init_val, bool enable_sharing);
        ~GpuFence() noexcept;

        GpuFence(GpuFence&& other) noexcept;
        GpuFence& operator=(GpuFence&& other) noexcept;

        explicit operator bool() const noexcept;

        void* NativeFence() const noexcept;
        template <typename Traits>
        typename Traits::FenceType NativeFence() const noexcept
        {
            return reinterpret_cast<typename Traits::FenceType>(this->NativeFence());
        }

        void* SharedFenceHandle() const noexcept;
        template <typename Traits>
        typename Traits::SharedHandleType SharedFenceHandle() const noexcept
        {
            return reinterpret_cast<typename Traits::SharedHandleType>(this->SharedFenceHandle());
        }

        uint64_t CompletedValue() const;

        void CpuWait(uint64_t value) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
