// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <chrono>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuCommandList.hpp"

namespace AIHoloImager
{
    class GpuTimerQueryInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTimerQueryInternal)

    public:
        GpuTimerQueryInternal() noexcept;
        virtual ~GpuTimerQueryInternal();

        GpuTimerQueryInternal(GpuTimerQueryInternal&& other) noexcept;
        virtual GpuTimerQueryInternal& operator=(GpuTimerQueryInternal&& other) noexcept = 0;

        virtual void Begin(GpuCommandList& cmd_list) = 0;
        virtual void End(GpuCommandList& cmd_list) = 0;

        virtual std::chrono::duration<double> Elapsed() const = 0;
    };
} // namespace AIHoloImager
