// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <chrono>
#include <future>
#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/InternalDefine.hpp"

namespace AIHoloImager
{
    class GpuCommandList;
    class GpuSystem;

    class GpuTimerQueryInternal;

    class GpuTimerQuery final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTimerQuery)
        DEFINE_INTERNAL(GpuTimerQuery)

    public:
        GpuTimerQuery() noexcept;
        explicit GpuTimerQuery(GpuSystem& gpu_system);
        ~GpuTimerQuery() noexcept;

        GpuTimerQuery(GpuTimerQuery&& other) noexcept;
        GpuTimerQuery& operator=(GpuTimerQuery&& other) noexcept;

        explicit operator bool() const noexcept;

        void Begin(GpuCommandList& cmd_list);
        void End(GpuCommandList& cmd_list);

        // The caller is responsible for ensuring that the command list has been executed and the GPU has completed processing the commands
        // before calling this method.
        std::chrono::duration<double> Elapsed() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
