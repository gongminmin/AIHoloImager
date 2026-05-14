// Copyright (c) 2026 Minmin Gong
//

#include "D3D12CommandQueue.hpp"

#include "Base/ErrorHandling.hpp"
#include "Base/Uuid.hpp"

#include "D3D12CommandList.hpp"
#include "D3D12Conversion.hpp"
#include "D3D12Fence.hpp"
#include "D3D12System.hpp"
#include "D3D12Util.hpp"

DEFINE_UUID_OF(ID3D12CommandQueue);

namespace AIHoloImager
{
    D3D12_IMP_IMP(CommandQueue)

    D3D12CommandQueue::D3D12CommandQueue() noexcept = default;
    D3D12CommandQueue::D3D12CommandQueue(GpuSystem& gpu_system, GpuSystem::CmdQueueType type, std::string_view name)
    {
        ID3D12Device* d3d12_device = D3D12Imp(gpu_system).Device();

        const D3D12_COMMAND_QUEUE_DESC queue_desc{
            .Type = ToD3D12CommandListType(type),
            .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
            .NodeMask = 0,
        };
        TIFHR(d3d12_device->CreateCommandQueue(&queue_desc, UuidOf<ID3D12CommandQueue>(), cmd_queue_.PutVoid()));
        SetName(*cmd_queue_, name);
    }
    D3D12CommandQueue::~D3D12CommandQueue() noexcept = default;

    D3D12CommandQueue::D3D12CommandQueue(D3D12CommandQueue&& other) noexcept = default;
    D3D12CommandQueue::D3D12CommandQueue(GpuCommandQueueInternal&& other) noexcept
        : D3D12CommandQueue(static_cast<D3D12CommandQueue&&>(other))
    {
    }
    D3D12CommandQueue& D3D12CommandQueue::operator=(D3D12CommandQueue&& other) noexcept = default;
    GpuCommandQueueInternal& D3D12CommandQueue::operator=(GpuCommandQueueInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12CommandQueue&&>(other));
    }

    D3D12CommandQueue::operator bool() const noexcept
    {
        return static_cast<bool>(cmd_queue_);
    }

    void* D3D12CommandQueue::NativeCmdQueue() const noexcept
    {
        return this->CmdQueue();
    }

    void D3D12CommandQueue::GpuWait(std::span<const GpuCommandQueue::FenceInfo> fences)
    {
        for (const auto& fence_info : fences)
        {
            TIFHR(cmd_queue_->Wait(D3D12Imp(*fence_info.fence).Fence(), fence_info.value));
        }
    }

    void D3D12CommandQueue::Execute(const GpuCommandList& cmd_list, std::span<const GpuCommandQueue::FenceInfo> wait_fences,
        const GpuCommandQueue::FenceInfo& signal_fence)
    {
        this->Execute(cmd_list.Internal(), std::move(wait_fences), signal_fence);
    }
    void D3D12CommandQueue::Execute(const GpuCommandListInternal& cmd_list_internal,
        std::span<const GpuCommandQueue::FenceInfo> wait_fences, const GpuCommandQueue::FenceInfo& signal_fence)
    {
        this->GpuWait(wait_fences);

        ID3D12CommandList* cmd_lists[] = {D3D12Imp(cmd_list_internal).CommandListBase()};
        cmd_queue_->ExecuteCommandLists(static_cast<uint32_t>(std::size(cmd_lists)), cmd_lists);

        TIFHR(cmd_queue_->Signal(D3D12Imp(*signal_fence.fence).Fence(), signal_fence.value));
    }

    ID3D12CommandQueue* D3D12CommandQueue::CmdQueue() const noexcept
    {
        return cmd_queue_.Get();
    }
} // namespace AIHoloImager
