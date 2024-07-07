// Copyright (c) 2024 Minmin Gong
//

#include "GpuCommandList.hpp"

#include <directx/d3d12.h>
#include <directx/d3d12video.h>

#include "GpuResourceViews.hpp"
#include "GpuSystem.hpp"
#include "Util/ErrorHandling.hpp"
#include "Util/Uuid.hpp"

DEFINE_UUID_OF(ID3D12GraphicsCommandList);
DEFINE_UUID_OF(ID3D12VideoEncodeCommandList);

namespace AIHoloImager
{
    GpuCommandList::GpuCommandList() noexcept = default;

    GpuCommandList::GpuCommandList(GpuSystem& gpu_system, ID3D12CommandAllocator* cmd_allocator, GpuSystem::CmdQueueType type)
        : gpu_system_(&gpu_system), type_(type)
    {
        ID3D12Device* d3d12_device = gpu_system.NativeDevice();
        switch (type)
        {
        case GpuSystem::CmdQueueType::Render:
            TIFHR(d3d12_device->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmd_allocator, nullptr, UuidOf<ID3D12GraphicsCommandList>(), cmd_list_.PutVoid()));
            break;

        case GpuSystem::CmdQueueType::Compute:
            TIFHR(d3d12_device->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_COMPUTE, cmd_allocator, nullptr, UuidOf<ID3D12GraphicsCommandList>(), cmd_list_.PutVoid()));
            break;

        case GpuSystem::CmdQueueType::VideoEncode:
            TIFHR(d3d12_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE, cmd_allocator, nullptr,
                UuidOf<ID3D12VideoEncodeCommandList>(), cmd_list_.PutVoid()));
            break;

        default:
            Unreachable();
        }
    }

    GpuCommandList::~GpuCommandList() noexcept = default;

    GpuCommandList::GpuCommandList(GpuCommandList&& other) noexcept = default;
    GpuCommandList& GpuCommandList::operator=(GpuCommandList&& other) noexcept = default;

    GpuSystem::CmdQueueType GpuCommandList::Type() const noexcept
    {
        return type_;
    }

    GpuCommandList::operator bool() const noexcept
    {
        return cmd_list_ ? true : false;
    }

    void GpuCommandList::Transition(std::span<const D3D12_RESOURCE_BARRIER> barriers) const noexcept
    {
        switch (type_)
        {
        case GpuSystem::CmdQueueType::Render:
        case GpuSystem::CmdQueueType::Compute:
            static_cast<ID3D12GraphicsCommandList*>(cmd_list_.Get())
                ->ResourceBarrier(static_cast<uint32_t>(barriers.size()), barriers.data());
            break;

        case GpuSystem::CmdQueueType::VideoEncode:
            static_cast<ID3D12VideoEncodeCommandList*>(cmd_list_.Get())
                ->ResourceBarrier(static_cast<uint32_t>(barriers.size()), barriers.data());
            break;

        default:
            Unreachable();
        }
    }

    void GpuCommandList::Compute(const GpuComputeShader& shader, uint32_t group_x, uint32_t group_y, uint32_t group_z,
        std::span<const GeneralConstantBuffer*> cbs, std::span<const GpuTexture2D*> srvs, std::span<GpuTexture2D*> uavs)
    {
        assert(gpu_system_ != nullptr);

        ID3D12GraphicsCommandList* d3d12_cmd_list;
        switch (type_)
        {
        case GpuSystem::CmdQueueType::Render:
        case GpuSystem::CmdQueueType::Compute:
            d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();
            break;

        default:
            throw std::runtime_error("This type of command list can't Compute.");
        }

        d3d12_cmd_list->SetPipelineState(shader.NativePipelineState());
        d3d12_cmd_list->SetComputeRootSignature(shader.NativeRootSignature());

        auto srv_uav_desc_block = gpu_system_->AllocCbvSrvUavDescBlock(2);
        const uint32_t srv_uav_desc_size = gpu_system_->CbvSrvUavDescSize();

        ID3D12DescriptorHeap* heaps[] = {srv_uav_desc_block.NativeDescriptorHeap()};
        d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);

        uint32_t heap_base = 0;
        uint32_t root_index = 0;

        d3d12_cmd_list->SetComputeRootDescriptorTable(
            root_index, OffsetHandle(srv_uav_desc_block.GpuHandle(), heap_base, srv_uav_desc_size));
        std::vector<GpuShaderResourceView> sr_views;
        for (uint32_t i = 0; i < static_cast<uint32_t>(srvs.size()); ++i, ++heap_base)
        {
            const auto* srv = srvs[i];
            if (srv != nullptr)
            {
                srv->Transition(*this, D3D12_RESOURCE_STATE_COMMON);
                sr_views.push_back(
                    GpuShaderResourceView(*gpu_system_, *srv, OffsetHandle(srv_uav_desc_block.CpuHandle(), heap_base, srv_uav_desc_size)));
            }
        }
        if (!srvs.empty())
        {
            ++root_index;
        }

        d3d12_cmd_list->SetComputeRootDescriptorTable(
            root_index, OffsetHandle(srv_uav_desc_block.GpuHandle(), heap_base, srv_uav_desc_size));
        std::vector<GpuUnorderedAccessView> ua_views;
        for (uint32_t i = 0; i < static_cast<uint32_t>(uavs.size()); ++i, ++heap_base)
        {
            const auto* uav = uavs[i];
            if (uav != nullptr)
            {
                uav->Transition(*this, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                ua_views.push_back(
                    GpuUnorderedAccessView(*gpu_system_, *uav, OffsetHandle(srv_uav_desc_block.CpuHandle(), heap_base, srv_uav_desc_size)));
            }
        }
        if (!uavs.empty())
        {
            ++root_index;
        }

        for (uint32_t i = 0; i < static_cast<uint32_t>(cbs.size()); ++i, ++root_index)
        {
            d3d12_cmd_list->SetComputeRootConstantBufferView(root_index, cbs[i]->GpuVirtualAddress());
        }

        d3d12_cmd_list->Dispatch(group_x, group_y, group_z);

        for (uint32_t i = 0; i < static_cast<uint32_t>(uavs.size()); ++i, ++heap_base)
        {
            const auto* uav = uavs[i];
            if (uav != nullptr)
            {
                uav->Transition(*this, D3D12_RESOURCE_STATE_COMMON);
            }
        }

        gpu_system_->DeallocCbvSrvUavDescBlock(std::move(srv_uav_desc_block));
    }

    void GpuCommandList::Close()
    {
        switch (type_)
        {
        case GpuSystem::CmdQueueType::Render:
        case GpuSystem::CmdQueueType::Compute:
            static_cast<ID3D12GraphicsCommandList*>(cmd_list_.Get())->Close();
            break;

        case GpuSystem::CmdQueueType::VideoEncode:
            static_cast<ID3D12VideoEncodeCommandList*>(cmd_list_.Get())->Close();
            break;

        default:
            Unreachable();
        }
    }

    void GpuCommandList::Reset(ID3D12CommandAllocator* cmd_allocator)
    {
        switch (type_)
        {
        case GpuSystem::CmdQueueType::Render:
        case GpuSystem::CmdQueueType::Compute:
            static_cast<ID3D12GraphicsCommandList*>(cmd_list_.Get())->Reset(cmd_allocator, nullptr);
            break;

        case GpuSystem::CmdQueueType::VideoEncode:
            static_cast<ID3D12VideoEncodeCommandList*>(cmd_list_.Get())->Reset(cmd_allocator);
            break;

        default:
            Unreachable();
        }
    }
} // namespace AIHoloImager
