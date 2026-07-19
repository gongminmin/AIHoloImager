// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <chrono>
#include <memory>

#include "Base/Noncopyable.hpp"
#include "Gpu/InternalDefine.hpp"
#include "Gpu/Symbol.hpp"

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
        AIHI_GPU_SYS_API GpuTimerQuery() noexcept;
        AIHI_GPU_SYS_API explicit GpuTimerQuery(GpuSystem& gpu_system);
        AIHI_GPU_SYS_API ~GpuTimerQuery() noexcept;

        AIHI_GPU_SYS_API GpuTimerQuery(GpuTimerQuery&& other) noexcept;
        AIHI_GPU_SYS_API GpuTimerQuery& operator=(GpuTimerQuery&& other) noexcept;

        AIHI_GPU_SYS_API explicit operator bool() const noexcept;

        AIHI_GPU_SYS_API void Begin(GpuCommandList& cmd_list);
        AIHI_GPU_SYS_API void End(GpuCommandList& cmd_list);

        // The caller is responsible for ensuring that the command list has been executed and the GPU has completed processing the commands
        // before calling this method.
        AIHI_GPU_SYS_API std::chrono::duration<double> Elapsed() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
