// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Gpu/GpuQuery.hpp"

#include "../GpuQueryInternal.hpp"
#include "D3D12ImpDefine.hpp"
#include "D3D12Util.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class D3D12TimerQuery : public GpuTimerQueryInternal
    {
    public:
        explicit D3D12TimerQuery(GpuSystem& gpu_system);
        ~D3D12TimerQuery() override;

        D3D12TimerQuery(D3D12TimerQuery&& other) noexcept;
        explicit D3D12TimerQuery(GpuTimerQueryInternal&& other) noexcept;

        D3D12TimerQuery& operator=(D3D12TimerQuery&& other) noexcept;
        D3D12TimerQuery& operator=(GpuTimerQueryInternal&& other) noexcept override;

        void Begin(GpuCommandList& cmd_list) override;
        void End(GpuCommandList& cmd_list) override;

        std::chrono::duration<double> Elapsed() const override;

    private:
        GpuSystem* gpu_system_ = nullptr;

        D3D12RecyclableObject<ComPtr<ID3D12QueryHeap>> timestamp_heap_;
        GpuMemoryBlock query_result_;
        uint64_t freq_ = 0;
    };

    D3D12_DEFINE_IMP(TimerQuery)
} // namespace AIHoloImager
