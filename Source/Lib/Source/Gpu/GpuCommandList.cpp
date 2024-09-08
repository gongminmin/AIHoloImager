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

    GpuCommandList::~GpuCommandList()
    {
        if (cmd_list_ && !closed_)
        {
            Unreachable("Command list is destructed without executing.");
        }
    }

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

    void GpuCommandList::Render(const GpuRenderPipeline& pipeline, std::span<const VertexBufferBinding> vbs, const IndexBufferBinding* ib,
        uint32_t num, const ShaderBinding shader_bindings[GpuRenderPipeline::NumShaderStages], std::span<const GpuRenderTargetView*> rtvs,
        const GpuDepthStencilView* dsv, std::span<const D3D12_VIEWPORT> viewports, std::span<const D3D12_RECT> scissor_rects)
    {
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

        if (!vbs.empty())
        {
            auto vbvs = std::make_unique<D3D12_VERTEX_BUFFER_VIEW[]>(vbs.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(vbs.size()); ++i)
            {
                const auto& vb_binding = vbs[i];
                assert(vb_binding.vb != nullptr);

                vb_binding.vb->Transition(*this, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

                D3D12_VERTEX_BUFFER_VIEW& vbv = vbvs[i];
                vbv.BufferLocation = vb_binding.vb->GpuVirtualAddress() + vb_binding.offset;
                vbv.SizeInBytes = vb_binding.vb->Size();
                vbv.StrideInBytes = vb_binding.stride;
            }
            d3d12_cmd_list->IASetVertexBuffers(0, static_cast<uint32_t>(vbs.size()), vbvs.get());
        }
        else
        {
            d3d12_cmd_list->IASetVertexBuffers(0, 0, nullptr);
        }

        if (ib != nullptr)
        {
            ib->ib->Transition(*this, D3D12_RESOURCE_STATE_INDEX_BUFFER);

            D3D12_INDEX_BUFFER_VIEW ibv;
            ibv.BufferLocation = ib->ib->GpuVirtualAddress() + ib->offset;
            ibv.SizeInBytes = ib->ib->Size();
            ibv.Format = ib->format;
            d3d12_cmd_list->IASetIndexBuffer(&ibv);
        }
        else
        {
            d3d12_cmd_list->IASetIndexBuffer(nullptr);
        }

        for (const auto* rtv : rtvs)
        {
            if (rtv != nullptr)
            {
                rtv->Transition(*this);
            }
        }
        if (dsv != nullptr)
        {
            dsv->Transition(*this);
        }

        d3d12_cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        d3d12_cmd_list->SetPipelineState(pipeline.NativePipelineState());
        d3d12_cmd_list->SetGraphicsRootSignature(pipeline.NativeRootSignature());

        uint32_t num_descs = 0;
        for (uint32_t s = 0; s < GpuRenderPipeline::NumShaderStages; ++s)
        {
            const auto& binding = shader_bindings[s];
            num_descs += static_cast<uint32_t>(binding.srvs.size() + binding.uavs.size());
        }

        GpuDescriptorBlock srv_uav_desc_block;
        if (num_descs > 0)
        {
            srv_uav_desc_block = gpu_system_->AllocShaderVisibleCbvSrvUavDescBlock(num_descs);

            ID3D12DescriptorHeap* heaps[] = {srv_uav_desc_block.NativeDescriptorHeap()};
            d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);
        }
        const uint32_t srv_uav_desc_size = gpu_system_->CbvSrvUavDescSize();

        uint32_t heap_base = 0;
        uint32_t root_index = 0;

        for (uint32_t s = 0; s < GpuRenderPipeline::NumShaderStages; ++s)
        {
            const auto& binding = shader_bindings[s];
            if (!binding.srvs.empty())
            {
                d3d12_cmd_list->SetGraphicsRootDescriptorTable(
                    root_index, OffsetHandle(srv_uav_desc_block.GpuHandle(), heap_base, srv_uav_desc_size));
                for (const auto* srv : binding.srvs)
                {
                    if (srv != nullptr)
                    {
                        srv->Transition(*this);

                        auto srv_cpu_handle = OffsetHandle(srv_uav_desc_block.CpuHandle(), heap_base, srv_uav_desc_size);
                        srv->CopyTo(srv_cpu_handle);
                    }

                    ++heap_base;
                }

                ++root_index;
            }

            if (!binding.uavs.empty())
            {
                d3d12_cmd_list->SetGraphicsRootDescriptorTable(
                    root_index, OffsetHandle(srv_uav_desc_block.GpuHandle(), heap_base, srv_uav_desc_size));
                for (const auto* uav : binding.uavs)
                {
                    if (uav != nullptr)
                    {
                        uav->Transition(*this);

                        auto uav_cpu_handle = OffsetHandle(srv_uav_desc_block.CpuHandle(), heap_base, srv_uav_desc_size);
                        uav->CopyTo(uav_cpu_handle);
                    }

                    ++heap_base;
                }

                ++root_index;
            }

            for (const auto* cb : binding.cbs)
            {
                d3d12_cmd_list->SetGraphicsRootConstantBufferView(root_index, cb->GpuVirtualAddress());
                ++root_index;
            }
        }

        std::unique_ptr<D3D12_CPU_DESCRIPTOR_HANDLE[]> rt_views;
        if (!rtvs.empty())
        {
            rt_views = std::make_unique<D3D12_CPU_DESCRIPTOR_HANDLE[]>(rtvs.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(rtvs.size()); ++i)
            {
                if (rtvs[i] != nullptr)
                {
                    rt_views[i] = rtvs[i]->CpuHandle();
                }
                else
                {
                    rt_views[i] = {~0ULL};
                }
            }
        }
        D3D12_CPU_DESCRIPTOR_HANDLE ds_view;
        if (dsv != nullptr)
        {
            ds_view = dsv->CpuHandle();
        }
        d3d12_cmd_list->OMSetRenderTargets(static_cast<uint32_t>(rtvs.size()), rt_views.get(), false, dsv != nullptr ? &ds_view : nullptr);

        d3d12_cmd_list->RSSetViewports(static_cast<uint32_t>(viewports.size()), viewports.data());
        d3d12_cmd_list->RSSetScissorRects(static_cast<uint32_t>(scissor_rects.size()), scissor_rects.data());

        if (ib != nullptr)
        {
            d3d12_cmd_list->DrawIndexedInstanced(num, 1, 0, 0, 0);
        }
        else
        {
            d3d12_cmd_list->DrawInstanced(num, 1, 0, 0);
        }

        gpu_system_->DeallocShaderVisibleCbvSrvUavDescBlock(std::move(srv_uav_desc_block));
    }

    void GpuCommandList::Compute(
        const GpuComputePipeline& pipeline, uint32_t group_x, uint32_t group_y, uint32_t group_z, const ShaderBinding& shader_binding)
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

        d3d12_cmd_list->SetPipelineState(pipeline.NativePipelineState());
        d3d12_cmd_list->SetComputeRootSignature(pipeline.NativeRootSignature());

        const uint32_t num_descs = static_cast<uint32_t>(shader_binding.srvs.size() + shader_binding.uavs.size());
        GpuDescriptorBlock srv_uav_desc_block;
        if (num_descs > 0)
        {
            srv_uav_desc_block = gpu_system_->AllocShaderVisibleCbvSrvUavDescBlock(num_descs);

            ID3D12DescriptorHeap* heaps[] = {srv_uav_desc_block.NativeDescriptorHeap()};
            d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);
        }
        const uint32_t srv_uav_desc_size = gpu_system_->CbvSrvUavDescSize();

        uint32_t heap_base = 0;
        uint32_t root_index = 0;

        if (!shader_binding.srvs.empty())
        {
            d3d12_cmd_list->SetComputeRootDescriptorTable(
                root_index, OffsetHandle(srv_uav_desc_block.GpuHandle(), heap_base, srv_uav_desc_size));
            for (const auto* srv : shader_binding.srvs)
            {
                if (srv != nullptr)
                {
                    srv->Transition(*this);

                    auto srv_cpu_handle = OffsetHandle(srv_uav_desc_block.CpuHandle(), heap_base, srv_uav_desc_size);
                    srv->CopyTo(srv_cpu_handle);
                }

                ++heap_base;
            }

            ++root_index;
        }

        if (!shader_binding.uavs.empty())
        {
            d3d12_cmd_list->SetComputeRootDescriptorTable(
                root_index, OffsetHandle(srv_uav_desc_block.GpuHandle(), heap_base, srv_uav_desc_size));
            for (const auto* uav : shader_binding.uavs)
            {
                if (uav != nullptr)
                {
                    uav->Transition(*this);

                    auto uav_cpu_handle = OffsetHandle(srv_uav_desc_block.CpuHandle(), heap_base, srv_uav_desc_size);
                    uav->CopyTo(uav_cpu_handle);
                }

                ++heap_base;
            }

            ++root_index;
        }

        for (const auto* cb : shader_binding.cbs)
        {
            d3d12_cmd_list->SetComputeRootConstantBufferView(root_index, cb->GpuVirtualAddress());
            ++root_index;
        }

        d3d12_cmd_list->Dispatch(group_x, group_y, group_z);

        gpu_system_->DeallocShaderVisibleCbvSrvUavDescBlock(std::move(srv_uav_desc_block));
    }

    void GpuCommandList::Copy(GpuBuffer& dest, const GpuBuffer& src)
    {
        ID3D12GraphicsCommandList* d3d12_cmd_list;
        switch (type_)
        {
        case GpuSystem::CmdQueueType::Render:
        case GpuSystem::CmdQueueType::Compute:
            d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();
            break;

        default:
            throw std::runtime_error("This type of command list can't Copy.");
        }

        src.Transition(*this, D3D12_RESOURCE_STATE_COPY_SOURCE);
        dest.Transition(*this, D3D12_RESOURCE_STATE_COPY_DEST);

        d3d12_cmd_list->CopyResource(dest.NativeBuffer(), src.NativeBuffer());
    }

    void GpuCommandList::Copy(GpuTexture2D& dest, const GpuTexture2D& src)
    {
        ID3D12GraphicsCommandList* d3d12_cmd_list;
        switch (type_)
        {
        case GpuSystem::CmdQueueType::Render:
        case GpuSystem::CmdQueueType::Compute:
            d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();
            break;

        default:
            throw std::runtime_error("This type of command list can't Copy.");
        }

        src.Transition(*this, D3D12_RESOURCE_STATE_COPY_SOURCE);
        dest.Transition(*this, D3D12_RESOURCE_STATE_COPY_DEST);

        d3d12_cmd_list->CopyResource(dest.NativeTexture(), src.NativeTexture());
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
        closed_ = true;
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
        closed_ = false;
    }
} // namespace AIHoloImager
