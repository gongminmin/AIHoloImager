// Copyright (c) 2025 Minmin Gong
//

#include "D3D12CommandAllocatorInfo.hpp"

#include "Base/Uuid.hpp"
#include "Gpu/D3D12/D3D12Traits.hpp"

namespace AIHoloImager
{
    D3D12CommandAllocatorInfo::D3D12CommandAllocatorInfo(GpuSystem& gpu_system, GpuSystem::CmdQueueType type)
    {
        D3D12_COMMAND_LIST_TYPE d3d12_type;
        switch (type)
        {
        case GpuSystem::CmdQueueType::Render:
            d3d12_type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            break;

        case GpuSystem::CmdQueueType::Compute:
            d3d12_type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            break;

        case GpuSystem::CmdQueueType::VideoEncode:
            d3d12_type = D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE;
            break;

        default:
            Unreachable();
        }

        ID3D12Device* d3d12_device = gpu_system.NativeDevice<D3D12Traits>();
        TIFHR(d3d12_device->CreateCommandAllocator(d3d12_type, UuidOf<ID3D12CommandAllocator>(), cmd_allocator_.PutVoid()));
    }
    D3D12CommandAllocatorInfo::~D3D12CommandAllocatorInfo() noexcept = default;

    D3D12CommandAllocatorInfo::D3D12CommandAllocatorInfo(D3D12CommandAllocatorInfo&& other) noexcept = default;
    D3D12CommandAllocatorInfo::D3D12CommandAllocatorInfo(GpuCommandAllocatorInfoInternal&& other) noexcept
        : D3D12CommandAllocatorInfo(static_cast<D3D12CommandAllocatorInfo&&>(other))
    {
    }
    D3D12CommandAllocatorInfo& D3D12CommandAllocatorInfo::operator=(D3D12CommandAllocatorInfo&& other) noexcept = default;
    GpuCommandAllocatorInfoInternal& D3D12CommandAllocatorInfo::operator=(GpuCommandAllocatorInfoInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12CommandAllocatorInfo&&>(other));
    }

    ID3D12CommandAllocator* D3D12CommandAllocatorInfo::NativeCmdAllocator() const noexcept
    {
        return cmd_allocator_.Get();
    }

    uint64_t D3D12CommandAllocatorInfo::FenceValue() const noexcept
    {
        return fence_val_;
    }

    void D3D12CommandAllocatorInfo::FenceValue(uint64_t value) noexcept
    {
        fence_val_ = value;
    }
} // namespace AIHoloImager
