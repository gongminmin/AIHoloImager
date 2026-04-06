// Copyright (c) 2026 Minmin Gong
//

#include "Gpu/GpuQuery.hpp"

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"

#include "Internal/GpuQueryInternal.hpp"
#include "Internal/GpuSystemInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    EMPTY_IMP(GpuTimerQuery)
    IMP_INTERNAL(GpuTimerQuery)

    GpuTimerQuery::GpuTimerQuery() noexcept = default;
    GpuTimerQuery::GpuTimerQuery(GpuSystem& gpu_system) : impl_(static_cast<Impl*>(gpu_system.Internal().CreateTimerQuery().release()))
    {
    }

    GpuTimerQuery::~GpuTimerQuery() noexcept = default;

    GpuTimerQuery::GpuTimerQuery(GpuTimerQuery&& other) noexcept = default;
    GpuTimerQuery& GpuTimerQuery::operator=(GpuTimerQuery&& other) noexcept = default;

    GpuTimerQuery::operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    void GpuTimerQuery::Begin(GpuCommandList& cmd_list)
    {
        impl_->Begin(cmd_list);
    }

    void GpuTimerQuery::End(GpuCommandList& cmd_list)
    {
        impl_->End(cmd_list);
    }

    std::chrono::duration<double> GpuTimerQuery::Elapsed() const
    {
        return impl_->Elapsed();
    }
} // namespace AIHoloImager
