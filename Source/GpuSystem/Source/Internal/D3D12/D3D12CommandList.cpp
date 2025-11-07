// Copyright (c) 2025 Minmin Gong
//

#include "D3D12CommandList.hpp"

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>
#include <directx/d3d12video.h>

#include "Base/ErrorHandling.hpp"
#include "Base/Uuid.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuSystem.hpp"

#include "D3D12Buffer.hpp"
#include "D3D12CommandAllocatorInfo.hpp"
#include "D3D12Conversion.hpp"
#include "D3D12DescriptorHeap.hpp"
#include "D3D12Resource.hpp"
#include "D3D12ResourceViews.hpp"
#include "D3D12Sampler.hpp"
#include "D3D12Shader.hpp"
#include "D3D12System.hpp"
#include "D3D12Texture.hpp"

DEFINE_UUID_OF(ID3D12GraphicsCommandList);
DEFINE_UUID_OF(ID3D12VideoEncodeCommandList);

namespace AIHoloImager
{
    D3D12_IMP_IMP(CommandList)

    D3D12CommandList::D3D12CommandList(GpuSystem& gpu_system, GpuCommandAllocatorInfo& cmd_alloc_info, GpuSystem::CmdQueueType type)
        : gpu_system_(&gpu_system), cmd_alloc_info_(&cmd_alloc_info), type_(type)
    {
        ID3D12Device* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto& d3d12_cmd_alloc_info = D3D12Imp(cmd_alloc_info);
        auto* cmd_allocator = d3d12_cmd_alloc_info.CmdAllocator();
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

        d3d12_cmd_alloc_info.RegisterAllocatedCommandList(cmd_list_.Get());
    }

    D3D12CommandList::~D3D12CommandList()
    {
        if (cmd_list_ && !closed_)
        {
            Unreachable("Command list is destructed without executing.");
        }
    }

    D3D12CommandList::D3D12CommandList(D3D12CommandList&& other) noexcept = default;
    D3D12CommandList::D3D12CommandList(GpuCommandListInternal&& other) noexcept : D3D12CommandList(static_cast<D3D12CommandList&&>(other))
    {
    }

    D3D12CommandList& D3D12CommandList::operator=(D3D12CommandList&& other) noexcept = default;
    GpuCommandListInternal& D3D12CommandList::operator=(GpuCommandListInternal&& other) noexcept
    {
        return this->operator=(static_cast<GpuCommandListInternal&&>(other));
    }

    GpuSystem::CmdQueueType D3D12CommandList::Type() const noexcept
    {
        return type_;
    }

    D3D12CommandList::operator bool() const noexcept
    {
        return cmd_list_ ? true : false;
    }

    template <>
    ID3D12GraphicsCommandList* D3D12CommandList::NativeCommandList<ID3D12GraphicsCommandList>() const
    {
        ID3D12GraphicsCommandList* d3d12_cmd_list;
        switch (type_)
        {
        case GpuSystem::CmdQueueType::Render:
        case GpuSystem::CmdQueueType::Compute:
            d3d12_cmd_list = static_cast<ID3D12GraphicsCommandList*>(cmd_list_.Get());
            break;

        default:
            throw std::runtime_error("This type of command list doesn't have a ID3D12GraphicsCommandList.");
        }
        return d3d12_cmd_list;
    }

    template <>
    ID3D12VideoEncodeCommandList* D3D12CommandList::NativeCommandList<ID3D12VideoEncodeCommandList>() const
    {
        ID3D12VideoEncodeCommandList* d3d12_cmd_list;
        switch (type_)
        {
        case GpuSystem::CmdQueueType::VideoEncode:
            d3d12_cmd_list = static_cast<ID3D12VideoEncodeCommandList*>(cmd_list_.Get());
            break;

        default:
            throw std::runtime_error("This type of command list doesn't have a ID3D12VideoEncodeCommandList.");
        }
        return d3d12_cmd_list;
    }

    void D3D12CommandList::Transition(std::span<const D3D12_RESOURCE_BARRIER> barriers) const noexcept
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

    void D3D12CommandList::Clear(GpuRenderTargetView& rtv, const float color[4])
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(rtv).Transition(*this);
        d3d12_cmd_list->ClearRenderTargetView(ToD3D12CpuDescriptorHandle(rtv.CpuHandle()), color, 0, nullptr);
    }

    void D3D12CommandList::Clear(GpuUnorderedAccessView& uav, const float color[4])
    {
        ID3D12Resource* resource = nullptr;
        GpuResource* uav_resource = uav.Resource();
        if (uav_resource)
        {
            resource = D3D12Imp(*uav_resource).Resource();
        }

        if (!resource)
        {
            return;
        }

        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(uav).Transition(*this);

        GpuDescriptorBlock uav_desc_block = gpu_system_->AllocShaderVisibleCbvSrvUavDescBlock(1);
        uav.CopyTo(uav_desc_block.CpuHandle());

        ID3D12DescriptorHeap* heaps[] = {D3D12Imp(*uav_desc_block.Heap()).DescriptorHeap()};
        d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);

        d3d12_cmd_list->ClearUnorderedAccessViewFloat(ToD3D12GpuDescriptorHandle(uav_desc_block.GpuHandle()),
            ToD3D12CpuDescriptorHandle(uav.CpuHandle()), resource, color, 0, nullptr);

        gpu_system_->DeallocShaderVisibleCbvSrvUavDescBlock(std::move(uav_desc_block));
    }

    void D3D12CommandList::Clear(GpuUnorderedAccessView& uav, const uint32_t color[4])
    {
        ID3D12Resource* resource = nullptr;
        GpuResource* uav_resource = uav.Resource();
        if (uav_resource)
        {
            resource = D3D12Imp(*uav_resource).Resource();
        }

        if (!resource)
        {
            return;
        }

        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(uav).Transition(*this);

        GpuDescriptorBlock uav_desc_block = gpu_system_->AllocShaderVisibleCbvSrvUavDescBlock(1);
        uav.CopyTo(uav_desc_block.CpuHandle());

        ID3D12DescriptorHeap* heaps[] = {D3D12Imp(*uav_desc_block.Heap()).DescriptorHeap()};
        d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);

        d3d12_cmd_list->ClearUnorderedAccessViewUint(ToD3D12GpuDescriptorHandle(uav_desc_block.GpuHandle()),
            ToD3D12CpuDescriptorHandle(uav.CpuHandle()), resource, color, 0, nullptr);

        gpu_system_->DeallocShaderVisibleCbvSrvUavDescBlock(std::move(uav_desc_block));
    }

    void D3D12CommandList::ClearDepth(GpuDepthStencilView& dsv, float depth)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(dsv).Transition(*this);
        d3d12_cmd_list->ClearDepthStencilView(ToD3D12CpuDescriptorHandle(dsv.CpuHandle()), D3D12_CLEAR_FLAG_DEPTH, depth, 0, 0, nullptr);
    }

    void D3D12CommandList::ClearStencil(GpuDepthStencilView& dsv, uint8_t stencil)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(dsv).Transition(*this);
        d3d12_cmd_list->ClearDepthStencilView(
            ToD3D12CpuDescriptorHandle(dsv.CpuHandle()), D3D12_CLEAR_FLAG_STENCIL, 0, stencil, 0, nullptr);
    }

    void D3D12CommandList::ClearDepthStencil(GpuDepthStencilView& dsv, float depth, uint8_t stencil)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(dsv).Transition(*this);
        d3d12_cmd_list->ClearDepthStencilView(
            ToD3D12CpuDescriptorHandle(dsv.CpuHandle()), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, depth, stencil, 0, nullptr);
    }

    void D3D12CommandList::Render(const GpuRenderPipeline& pipeline, std::span<const GpuCommandList::VertexBufferBinding> vbs,
        const GpuCommandList::IndexBufferBinding* ib, uint32_t num, std::span<const GpuCommandList::ShaderBinding> shader_bindings,
        std::span<const GpuRenderTargetView*> rtvs, const GpuDepthStencilView* dsv, std::span<const GpuViewport> viewports,
        std::span<const GpuRect> scissor_rects)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        if (!vbs.empty())
        {
            auto vbvs = std::make_unique<D3D12_VERTEX_BUFFER_VIEW[]>(vbs.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(vbs.size()); ++i)
            {
                const auto& vb_binding = vbs[i];
                assert(vb_binding.vb != nullptr);

                D3D12Imp(*vb_binding.vb).Transition(*this, GpuResourceState::Common);

                D3D12_VERTEX_BUFFER_VIEW& vbv = vbvs[i];
                vbv = {
                    .BufferLocation = vb_binding.vb->GpuVirtualAddress() + vb_binding.offset,
                    .SizeInBytes = vb_binding.vb->Size(),
                    .StrideInBytes = vb_binding.stride,
                };
            }
            d3d12_cmd_list->IASetVertexBuffers(0, static_cast<uint32_t>(vbs.size()), vbvs.get());
        }
        else
        {
            d3d12_cmd_list->IASetVertexBuffers(0, 0, nullptr);
        }

        if (ib != nullptr)
        {
            D3D12Imp(*ib->ib).Transition(*this, GpuResourceState::Common);

            const D3D12_INDEX_BUFFER_VIEW ibv{
                .BufferLocation = ib->ib->GpuVirtualAddress() + ib->offset,
                .SizeInBytes = ib->ib->Size(),
                .Format = ToDxgiFormat(ib->format),
            };
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
                D3D12Imp(*rtv).Transition(*this);
            }
        }
        if (dsv != nullptr)
        {
            D3D12Imp(*dsv).Transition(*this);
        }

        D3D12Imp(pipeline).Bind(*this);

        uint32_t num_srv_uav_descs = 0;
        uint32_t num_sampler_descs = 0;
        for (const auto& binding : shader_bindings)
        {
            num_srv_uav_descs += static_cast<uint32_t>(binding.srvs.size() + binding.uavs.size());
            num_sampler_descs += static_cast<uint32_t>(binding.samplers.size());
        }

        ID3D12DescriptorHeap* heaps[2] = {};
        uint32_t num_heaps = 0;

        GpuDescriptorBlock srv_uav_desc_block;
        if (num_srv_uav_descs > 0)
        {
            srv_uav_desc_block = gpu_system_->AllocShaderVisibleCbvSrvUavDescBlock(num_srv_uav_descs);
            heaps[num_heaps] = D3D12Imp(*srv_uav_desc_block.Heap()).DescriptorHeap();
            ++num_heaps;
        }
        GpuDescriptorBlock sampler_desc_block;
        if (num_sampler_descs > 0)
        {
            sampler_desc_block = gpu_system_->AllocShaderVisibleSamplerDescBlock(num_sampler_descs);
            heaps[num_heaps] = D3D12Imp(*sampler_desc_block.Heap()).DescriptorHeap();
            ++num_heaps;
        }
        if (num_heaps > 0)
        {
            d3d12_cmd_list->SetDescriptorHeaps(num_heaps, heaps);
        }

        const uint32_t srv_uav_desc_size = gpu_system_->CbvSrvUavDescSize();
        const uint32_t sampler_desc_size = gpu_system_->SamplerDescSize();

        uint32_t heap_base = 0;
        uint32_t root_index = 0;

        for (const auto& binding : shader_bindings)
        {
            if (!binding.srvs.empty())
            {
                d3d12_cmd_list->SetGraphicsRootDescriptorTable(
                    root_index, ToD3D12GpuDescriptorHandle(OffsetHandle(srv_uav_desc_block.GpuHandle(), heap_base, srv_uav_desc_size)));
                for (const auto* srv : binding.srvs)
                {
                    if (srv != nullptr)
                    {
                        D3D12Imp(*srv).Transition(*this);

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
                    root_index, ToD3D12GpuDescriptorHandle(OffsetHandle(srv_uav_desc_block.GpuHandle(), heap_base, srv_uav_desc_size)));
                for (const auto* uav : binding.uavs)
                {
                    if (uav != nullptr)
                    {
                        D3D12Imp(*uav).Transition(*this);

                        auto uav_cpu_handle = OffsetHandle(srv_uav_desc_block.CpuHandle(), heap_base, srv_uav_desc_size);
                        uav->CopyTo(uav_cpu_handle);
                    }

                    ++heap_base;
                }

                ++root_index;
            }

            heap_base = 0;
            if (!binding.samplers.empty())
            {
                d3d12_cmd_list->SetGraphicsRootDescriptorTable(
                    root_index, ToD3D12GpuDescriptorHandle(OffsetHandle(sampler_desc_block.GpuHandle(), heap_base, sampler_desc_size)));
                for (const auto* sampler : binding.samplers)
                {
                    if (sampler != nullptr)
                    {
                        auto sampler_cpu_handle = OffsetHandle(sampler_desc_block.CpuHandle(), heap_base, sampler_desc_size);
                        D3D12Imp(*sampler).CopyTo(sampler_cpu_handle);
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
                    rt_views[i] = ToD3D12CpuDescriptorHandle(rtvs[i]->CpuHandle());
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
            ds_view = ToD3D12CpuDescriptorHandle(dsv->CpuHandle());
        }
        d3d12_cmd_list->OMSetRenderTargets(static_cast<uint32_t>(rtvs.size()), rt_views.get(), false, dsv != nullptr ? &ds_view : nullptr);

        auto d3d12_viewports = std::make_unique<D3D12_VIEWPORT[]>(viewports.size());
        for (size_t i = 0; i < viewports.size(); ++i)
        {
            d3d12_viewports[i].TopLeftX = viewports[i].left;
            d3d12_viewports[i].TopLeftY = viewports[i].top;
            d3d12_viewports[i].Width = viewports[i].width;
            d3d12_viewports[i].Height = viewports[i].height;
            d3d12_viewports[i].MinDepth = viewports[i].min_depth;
            d3d12_viewports[i].MaxDepth = viewports[i].max_depth;
        }
        d3d12_cmd_list->RSSetViewports(static_cast<uint32_t>(viewports.size()), d3d12_viewports.get());

        if (scissor_rects.empty())
        {
            const D3D12_RECT d3d12_scissor_rect{
                .left = static_cast<LONG>(viewports[0].left),
                .top = static_cast<LONG>(viewports[0].top),
                .right = static_cast<LONG>(viewports[0].left + viewports[0].width),
                .bottom = static_cast<LONG>(viewports[0].top + viewports[0].height),
            };
            d3d12_cmd_list->RSSetScissorRects(1, &d3d12_scissor_rect);
        }
        else
        {
            auto d3d12_scissor_rects = std::make_unique<D3D12_RECT[]>(scissor_rects.size());
            for (size_t i = 0; i < scissor_rects.size(); ++i)
            {
                d3d12_scissor_rects[i] = {
                    .left = scissor_rects[i].left,
                    .top = scissor_rects[i].top,
                    .right = scissor_rects[i].right,
                    .bottom = scissor_rects[i].bottom,
                };
            }
            d3d12_cmd_list->RSSetScissorRects(static_cast<uint32_t>(scissor_rects.size()), d3d12_scissor_rects.get());
        }

        if (ib != nullptr)
        {
            d3d12_cmd_list->DrawIndexedInstanced(num, 1, 0, 0, 0);
        }
        else
        {
            d3d12_cmd_list->DrawInstanced(num, 1, 0, 0);
        }

        gpu_system_->DeallocShaderVisibleCbvSrvUavDescBlock(std::move(srv_uav_desc_block));
        gpu_system_->DeallocShaderVisibleSamplerDescBlock(std::move(sampler_desc_block));
    }

    void D3D12CommandList::Compute(const GpuComputePipeline& pipeline, uint32_t group_x, uint32_t group_y, uint32_t group_z,
        const GpuCommandList::ShaderBinding& shader_binding)
    {
        this->Compute(pipeline, shader_binding, [this, group_x, group_y, group_z]() {
            auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();
            d3d12_cmd_list->Dispatch(group_x, group_y, group_z);
        });
    }

    void D3D12CommandList::ComputeIndirect(
        const GpuComputePipeline& pipeline, const GpuBuffer& indirect_args, const GpuCommandList::ShaderBinding& shader_binding)
    {
        this->Compute(pipeline, shader_binding, [this, &indirect_args]() {
            const auto& d3d12_indirect_args = D3D12Imp(indirect_args);
            d3d12_indirect_args.Transition(*this, GpuResourceState::Common);

            auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();
            d3d12_cmd_list->ExecuteIndirect(
                D3D12Imp(*gpu_system_).NativeDispatchIndirectSignature(), 1, d3d12_indirect_args.Resource(), 0, nullptr, 0);
        });
    }

    void D3D12CommandList::Compute(
        const GpuComputePipeline& pipeline, const GpuCommandList::ShaderBinding& shader_binding, std::function<void()> dispatch_call)
    {
        assert(gpu_system_ != nullptr);

        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(pipeline).Bind(*this);

        const uint32_t num_srv_uav_descs = static_cast<uint32_t>(shader_binding.srvs.size() + shader_binding.uavs.size());
        const uint32_t num_sampler_descs = static_cast<uint32_t>(shader_binding.samplers.size());

        ID3D12DescriptorHeap* heaps[2] = {};
        uint32_t num_heaps = 0;

        GpuDescriptorBlock srv_uav_desc_block;
        if (num_srv_uav_descs > 0)
        {
            srv_uav_desc_block = gpu_system_->AllocShaderVisibleCbvSrvUavDescBlock(num_srv_uav_descs);
            heaps[num_heaps] = D3D12Imp(*srv_uav_desc_block.Heap()).DescriptorHeap();
            ++num_heaps;
        }
        GpuDescriptorBlock sampler_desc_block;
        if (num_sampler_descs > 0)
        {
            sampler_desc_block = gpu_system_->AllocShaderVisibleSamplerDescBlock(num_sampler_descs);
            heaps[num_heaps] = D3D12Imp(*sampler_desc_block.Heap()).DescriptorHeap();
            ++num_heaps;
        }
        if (num_heaps > 0)
        {
            d3d12_cmd_list->SetDescriptorHeaps(num_heaps, heaps);
        }

        const uint32_t srv_uav_desc_size = gpu_system_->CbvSrvUavDescSize();
        const uint32_t sampler_desc_size = gpu_system_->SamplerDescSize();

        uint32_t heap_base = 0;
        uint32_t root_index = 0;

        if (!shader_binding.srvs.empty())
        {
            d3d12_cmd_list->SetComputeRootDescriptorTable(
                root_index, ToD3D12GpuDescriptorHandle(OffsetHandle(srv_uav_desc_block.GpuHandle(), heap_base, srv_uav_desc_size)));
            for (const auto* srv : shader_binding.srvs)
            {
                if (srv != nullptr)
                {
                    D3D12Imp(*srv).Transition(*this);

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
                root_index, ToD3D12GpuDescriptorHandle(OffsetHandle(srv_uav_desc_block.GpuHandle(), heap_base, srv_uav_desc_size)));
            for (const auto* uav : shader_binding.uavs)
            {
                if (uav != nullptr)
                {
                    D3D12Imp(*uav).Transition(*this);

                    auto uav_cpu_handle = OffsetHandle(srv_uav_desc_block.CpuHandle(), heap_base, srv_uav_desc_size);
                    uav->CopyTo(uav_cpu_handle);
                }

                ++heap_base;
            }

            ++root_index;
        }

        heap_base = 0;
        if (!shader_binding.samplers.empty())
        {
            d3d12_cmd_list->SetComputeRootDescriptorTable(
                root_index, ToD3D12GpuDescriptorHandle(OffsetHandle(sampler_desc_block.GpuHandle(), heap_base, sampler_desc_size)));
            for (const auto* sampler : shader_binding.samplers)
            {
                if (sampler != nullptr)
                {
                    auto sampler_cpu_handle = OffsetHandle(sampler_desc_block.CpuHandle(), heap_base, sampler_desc_size);
                    D3D12Imp(*sampler).CopyTo(sampler_cpu_handle);
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

        dispatch_call();

        gpu_system_->DeallocShaderVisibleCbvSrvUavDescBlock(std::move(srv_uav_desc_block));
        gpu_system_->DeallocShaderVisibleSamplerDescBlock(std::move(sampler_desc_block));
    }

    void D3D12CommandList::Copy(GpuBuffer& dest, const GpuBuffer& src)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(src).Transition(*this, GpuResourceState::CopySrc);
        D3D12Imp(dest).Transition(*this, GpuResourceState::CopyDst);

        d3d12_cmd_list->CopyResource(D3D12Imp(dest).Resource(), D3D12Imp(src).Resource());
    }

    void D3D12CommandList::Copy(GpuBuffer& dest, uint32_t dst_offset, const GpuBuffer& src, uint32_t src_offset, uint32_t src_size)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(src).Transition(*this, GpuResourceState::CopySrc);
        D3D12Imp(dest).Transition(*this, GpuResourceState::CopyDst);

        d3d12_cmd_list->CopyBufferRegion(D3D12Imp(dest).Resource(), dst_offset, D3D12Imp(src).Resource(), src_offset, src_size);
    }

    void D3D12CommandList::Copy(GpuTexture& dest, const GpuTexture& src)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(src).Transition(*this, GpuResourceState::CopySrc);
        D3D12Imp(dest).Transition(*this, GpuResourceState::CopyDst);

        d3d12_cmd_list->CopyResource(D3D12Imp(dest).Resource(), D3D12Imp(src).Resource());
    }

    void D3D12CommandList::Copy(GpuTexture& dest, uint32_t dest_sub_resource, uint32_t dst_x, uint32_t dst_y, uint32_t dst_z,
        const GpuTexture& src, uint32_t src_sub_resource, const GpuBox& src_box)
    {
        const D3D12_TEXTURE_COPY_LOCATION src_loc{
            .pResource = D3D12Imp(src).Resource(),
            .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            .SubresourceIndex = src_sub_resource,
        };

        const D3D12_TEXTURE_COPY_LOCATION dst_loc{
            .pResource = D3D12Imp(dest).Resource(),
            .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            .SubresourceIndex = dest_sub_resource,
        };

        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(src).Transition(*this, GpuResourceState::CopySrc);
        D3D12Imp(dest).Transition(*this, GpuResourceState::CopyDst);

        const D3D12_BOX d3d12_src_box{
            .left = src_box.left,
            .top = src_box.top,
            .front = src_box.front,
            .right = src_box.right,
            .bottom = src_box.bottom,
            .back = src_box.back,
        };
        d3d12_cmd_list->CopyTextureRegion(&dst_loc, dst_x, dst_y, dst_z, &src_loc, &d3d12_src_box);
    }

    void D3D12CommandList::Upload(GpuBuffer& dest, const std::function<void(void* dst_data)>& copy_func)
    {
        auto* d3d12_dest = D3D12Imp(dest).Resource();

        D3D12_HEAP_PROPERTIES heap_prop;
        d3d12_dest->GetHeapProperties(&heap_prop, nullptr);

        switch (heap_prop.Type)
        {
        case D3D12_HEAP_TYPE_UPLOAD:
        case D3D12_HEAP_TYPE_READBACK:
            copy_func(dest.Map());
            dest.Unmap();
            break;

        case D3D12_HEAP_TYPE_DEFAULT:
        {
            auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

            auto upload_mem_block = gpu_system_->AllocUploadMemBlock(dest.Size(), gpu_system_->StructuredDataAlignment());

            void* buff_data = upload_mem_block.CpuSpan<std::byte>().data();
            copy_func(buff_data);

            D3D12Imp(dest).Transition(*this, GpuResourceState::CopyDst);
            d3d12_cmd_list->CopyBufferRegion(
                d3d12_dest, 0, D3D12Imp(*upload_mem_block.Buffer()).Resource(), upload_mem_block.Offset(), dest.Size());

            gpu_system_->DeallocUploadMemBlock(std::move(upload_mem_block));
        }
        break;

        default:
            Unreachable("Invalid heap type");
        }
    }

    void D3D12CommandList::Upload(GpuTexture& dest, uint32_t sub_resource,
        const std::function<void(void* dst_data, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func)
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, dest.MipLevels(), dest.ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = dest.Width(mip);
        const uint32_t height = dest.Height(mip);
        const uint32_t depth = dest.Depth(mip);

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_dest_texture = D3D12Imp(dest).Resource();

        const auto desc = d3d12_dest_texture->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
        uint64_t required_size = 0;
        d3d12_device->GetCopyableFootprints(&desc, sub_resource, 1, 0, &layout, nullptr, nullptr, &required_size);
        assert(layout.Footprint.RowPitch >= width * FormatSize(dest.Format()));

        auto upload_mem_block = gpu_system_->AllocUploadMemBlock(static_cast<uint32_t>(required_size), gpu_system_->TextureDataAlignment());

        void* tex_data = upload_mem_block.CpuSpan<std::byte>().data();
        copy_func(tex_data, layout.Footprint.RowPitch, layout.Footprint.RowPitch * layout.Footprint.Height);

        layout.Offset += upload_mem_block.Offset();
        const D3D12_TEXTURE_COPY_LOCATION src_loc{
            .pResource = D3D12Imp(*upload_mem_block.Buffer()).Resource(),
            .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            .PlacedFootprint = layout,
        };

        const D3D12_TEXTURE_COPY_LOCATION dst_loc{
            .pResource = d3d12_dest_texture,
            .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            .SubresourceIndex = sub_resource,
        };

        const D3D12_BOX src_box{
            .left = 0,
            .top = 0,
            .front = 0,
            .right = width,
            .bottom = height,
            .back = depth,
        };

        assert((type_ == GpuSystem::CmdQueueType::Render) || (type_ == GpuSystem::CmdQueueType::Compute));
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(dest).Transition(*this, GpuResourceState::CopyDst);

        d3d12_cmd_list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, &src_box);

        gpu_system_->DeallocUploadMemBlock(std::move(upload_mem_block));
    }

    std::future<void> D3D12CommandList::ReadBackAsync(const GpuBuffer& src, const std::function<void(const void* dst_data)>& copy_func)
    {
        auto* d3d12_src_buff = D3D12Imp(src).Resource();

        D3D12_HEAP_PROPERTIES heap_prop;
        d3d12_src_buff->GetHeapProperties(&heap_prop, nullptr);

        switch (heap_prop.Type)
        {
        case D3D12_HEAP_TYPE_UPLOAD:
        case D3D12_HEAP_TYPE_READBACK:
            copy_func(src.Map());
            src.Unmap();
            return {};

        case D3D12_HEAP_TYPE_DEFAULT:
        {
            auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

            auto read_back_mem_block = gpu_system_->AllocReadBackMemBlock(src.Size(), gpu_system_->StructuredDataAlignment());

            D3D12Imp(src).Transition(*this, GpuResourceState::CopySrc);

            d3d12_cmd_list->CopyBufferRegion(
                D3D12Imp(*read_back_mem_block.Buffer()).Resource(), read_back_mem_block.Offset(), d3d12_src_buff, 0, src.Size());
            const uint64_t fence_val = D3D12Imp(*gpu_system_).ExecuteAndReset(*this, GpuSystem::MaxFenceValue);

            return std::async(
                std::launch::deferred, [this, fence_val, read_back_mem_block = std::move(read_back_mem_block), copy_func]() mutable {
                    gpu_system_->CpuWait(fence_val);

                    const void* buff_data = read_back_mem_block.CpuSpan<std::byte>().data();
                    copy_func(buff_data);

                    gpu_system_->DeallocReadBackMemBlock(std::move(read_back_mem_block));
                });
        }

        default:
            Unreachable("Invalid heap type");
        }
    }

    std::future<void> D3D12CommandList::ReadBackAsync(const GpuTexture& src, uint32_t sub_resource,
        const std::function<void(const void* src_data, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func)
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, src.MipLevels(), src.ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = src.Width(mip);
        const uint32_t height = src.Height(mip);
        const uint32_t depth = src.Depth(mip);

        auto* d3d12_device = D3D12Imp(*gpu_system_).Device();
        auto* d3d12_src_texture = D3D12Imp(src).Resource();

        const auto desc = d3d12_src_texture->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
        uint64_t required_size = 0;
        d3d12_device->GetCopyableFootprints(&desc, sub_resource, 1, 0, &layout, nullptr, nullptr, &required_size);
        assert(layout.Footprint.RowPitch >= width * FormatSize(src.Format()));

        auto read_back_mem_block =
            gpu_system_->AllocReadBackMemBlock(static_cast<uint32_t>(required_size), gpu_system_->TextureDataAlignment());

        const D3D12_TEXTURE_COPY_LOCATION src_loc{
            .pResource = d3d12_src_texture,
            .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            .SubresourceIndex = sub_resource,
        };

        layout.Offset = read_back_mem_block.Offset();
        const D3D12_TEXTURE_COPY_LOCATION dst_loc{
            .pResource = D3D12Imp(*read_back_mem_block.Buffer()).Resource(),
            .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            .PlacedFootprint = layout,
        };

        const D3D12_BOX src_box{
            .left = 0,
            .top = 0,
            .front = 0,
            .right = width,
            .bottom = height,
            .back = depth,
        };

        assert((type_ == GpuSystem::CmdQueueType::Render) || (type_ == GpuSystem::CmdQueueType::Compute));
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        D3D12Imp(src).Transition(*this, GpuResourceState::CopySrc);

        d3d12_cmd_list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, &src_box);
        const uint64_t fence_val = D3D12Imp(*gpu_system_).ExecuteAndReset(*this, GpuSystem::MaxFenceValue);

        return std::async(std::launch::deferred,
            [this, fence_val, read_back_mem_block = std::move(read_back_mem_block), row_pitch = layout.Footprint.RowPitch,
                slice_pitch = layout.Footprint.RowPitch * layout.Footprint.Height, copy_func]() mutable {
                gpu_system_->CpuWait(fence_val);

                const void* tex_data = read_back_mem_block.CpuSpan<std::byte>().data();
                copy_func(tex_data, row_pitch, slice_pitch);

                gpu_system_->DeallocReadBackMemBlock(std::move(read_back_mem_block));
            });
    }

    void D3D12CommandList::Close()
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
        D3D12Imp(*cmd_alloc_info_).UnregisterAllocatedCommandList(cmd_list_.Get());
        closed_ = true;
        cmd_alloc_info_ = nullptr;
    }

    void D3D12CommandList::Reset(GpuCommandAllocatorInfo& cmd_alloc_info)
    {
        cmd_alloc_info_ = &cmd_alloc_info;
        auto& d3d12_cmd_alloc_info = D3D12Imp(cmd_alloc_info);
        auto* d3d12_cmd_alloc = d3d12_cmd_alloc_info.CmdAllocator();
        switch (type_)
        {
        case GpuSystem::CmdQueueType::Render:
        case GpuSystem::CmdQueueType::Compute:
            static_cast<ID3D12GraphicsCommandList*>(cmd_list_.Get())->Reset(d3d12_cmd_alloc, nullptr);
            break;

        case GpuSystem::CmdQueueType::VideoEncode:
            static_cast<ID3D12VideoEncodeCommandList*>(cmd_list_.Get())->Reset(d3d12_cmd_alloc);
            break;

        default:
            Unreachable();
        }
        d3d12_cmd_alloc_info.RegisterAllocatedCommandList(cmd_list_.Get());
        closed_ = false;
    }
} // namespace AIHoloImager
