// Copyright (c) 2026 Minmin Gong
//

#include "D3D12Query.hpp"

#include "Base/ErrorHandling.hpp"

#include "D3D12Buffer.hpp"
#include "D3D12CommandList.hpp"
#include "D3D12System.hpp"

DEFINE_UUID_OF(ID3D12QueryHeap);

namespace AIHoloImager
{
    D3D12TimerQuery::D3D12TimerQuery(GpuSystem& gpu_system) : gpu_system_(&gpu_system), timestamp_heap_(D3D12Imp(gpu_system), nullptr)
    {
        auto& d3d12_system = *timestamp_heap_.D3D12Sys();
        ID3D12Device* d3d12_device = d3d12_system.Device();

        D3D12_QUERY_HEAP_DESC desc{
            .Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
            .Count = 2,
            .NodeMask = 0,
        };
        TIFHR(d3d12_device->CreateQueryHeap(&desc, UuidOf<ID3D12QueryHeap>(), timestamp_heap_.Object().PutVoid()));

        query_result_ = gpu_system_->AllocReadBackMemBlock(sizeof(uint64_t) * 2, gpu_system_->StructuredDataAlignment());
    }

    D3D12TimerQuery::~D3D12TimerQuery()
    {
        if (query_result_)
        {
            gpu_system_->DeallocReadBackMemBlock(std::move(query_result_));
        }
    }

    D3D12TimerQuery::D3D12TimerQuery(D3D12TimerQuery&& other) noexcept = default;
    D3D12TimerQuery::D3D12TimerQuery(GpuTimerQueryInternal&& other) noexcept : D3D12TimerQuery(static_cast<D3D12TimerQuery&&>(other))
    {
    }

    D3D12TimerQuery& D3D12TimerQuery::operator=(D3D12TimerQuery&& other) noexcept = default;
    D3D12TimerQuery& D3D12TimerQuery::operator=(GpuTimerQueryInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12TimerQuery&&>(other));
    }

    void D3D12TimerQuery::Begin(GpuCommandList& cmd_list)
    {
        D3D12Imp(cmd_list).NativeCommandList<ID3D12GraphicsCommandList>()->EndQuery(
            timestamp_heap_.Object().Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }

    void D3D12TimerQuery::End(GpuCommandList& cmd_list)
    {
        auto& d3d12_cmd_list = D3D12Imp(cmd_list);

        d3d12_cmd_list.NativeCommandList<ID3D12GraphicsCommandList>()->EndQuery(
            timestamp_heap_.Object().Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);

        d3d12_cmd_list.RegisterAccessedObject(query_result_.StalledWaitFences());

        d3d12_cmd_list.NativeCommandList<ID3D12GraphicsCommandList>()->ResolveQueryData(timestamp_heap_.Object().Get(),
            D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, D3D12Imp(*query_result_.Buffer()).Resource(), query_result_.Offset());

        D3D12Imp(*gpu_system_).CommandQueue(cmd_list.Type())->GetTimestampFrequency(&freq_);
    }

    std::chrono::duration<double> D3D12TimerQuery::Elapsed() const
    {
        const auto timestamps = query_result_.CpuSpan<uint64_t>();
        const double seconds = static_cast<double>(timestamps[1] - timestamps[0]) / freq_;
        return std::chrono::duration<double>(seconds);
    }
} // namespace AIHoloImager
