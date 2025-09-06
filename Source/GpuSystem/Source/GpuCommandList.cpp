// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuCommandList.hpp"

#include <directx/d3d12.h>
#include <directx/d3d12video.h>

#include "Base/ErrorHandling.hpp"
#include "Base/Uuid.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuSystem.hpp"

DEFINE_UUID_OF(ID3D12GraphicsCommandList);
DEFINE_UUID_OF(ID3D12VideoEncodeCommandList);

namespace AIHoloImager
{
    GpuCommandList::GpuCommandList() noexcept = default;

    GpuCommandList::GpuCommandList(GpuSystem& gpu_system, GpuCommandAllocatorInfo& cmd_alloc_info, GpuSystem::CmdQueueType type)
        : gpu_system_(&gpu_system), cmd_alloc_info_(&cmd_alloc_info), type_(type)
    {
        ID3D12Device* d3d12_device = gpu_system.NativeDevice();
        auto* cmd_allocator = cmd_alloc_info.cmd_allocator.Get();
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

    template <>
    ID3D12GraphicsCommandList* GpuCommandList::NativeCommandList<ID3D12GraphicsCommandList>() const
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
    ID3D12VideoEncodeCommandList* GpuCommandList::NativeCommandList<ID3D12VideoEncodeCommandList>() const
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

    void GpuCommandList::Clear(GpuRenderTargetView& rtv, const float color[4])
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        rtv.Transition(*this);
        d3d12_cmd_list->ClearRenderTargetView(rtv.CpuHandle(), color, 0, nullptr);
    }

    void GpuCommandList::Clear(GpuUnorderedAccessView& uav, const float color[4])
    {
        ID3D12Resource* resource = nullptr;
        if (auto* tex_2d = uav.Texture2D())
        {
            resource = tex_2d->NativeTexture();
        }
        else if (auto* tex_2d_array = uav.Texture2DArray())
        {
            resource = tex_2d_array->NativeTexture();
        }
        else if (auto* tex_3d = uav.Texture3D())
        {
            resource = tex_3d->NativeTexture();
        }
        else if (auto* buff = uav.Buffer())
        {
            resource = buff->NativeBuffer();
        }

        if (!resource)
        {
            return;
        }

        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        uav.Transition(*this);

        GpuDescriptorBlock uav_desc_block = gpu_system_->AllocShaderVisibleCbvSrvUavDescBlock(1);
        uav.CopyTo(uav_desc_block.CpuHandle());

        ID3D12DescriptorHeap* heaps[] = {uav_desc_block.NativeDescriptorHeap()};
        d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);

        d3d12_cmd_list->ClearUnorderedAccessViewFloat(uav_desc_block.GpuHandle(), uav.CpuHandle(), resource, color, 0, nullptr);

        gpu_system_->DeallocShaderVisibleCbvSrvUavDescBlock(std::move(uav_desc_block));
    }

    void GpuCommandList::Clear(GpuUnorderedAccessView& uav, const uint32_t color[4])
    {
        ID3D12Resource* resource = nullptr;
        if (auto* tex_2d = uav.Texture2D())
        {
            resource = tex_2d->NativeTexture();
        }
        else if (auto* tex_2d_array = uav.Texture2D())
        {
            resource = tex_2d_array->NativeTexture();
        }
        else if (auto* tex_3d = uav.Texture3D())
        {
            resource = tex_3d->NativeTexture();
        }
        else if (auto* buff = uav.Buffer())
        {
            resource = buff->NativeBuffer();
        }


        if (!resource)
        {
            return;
        }

        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        uav.Transition(*this);

        GpuDescriptorBlock uav_desc_block = gpu_system_->AllocShaderVisibleCbvSrvUavDescBlock(1);
        uav.CopyTo(uav_desc_block.CpuHandle());

        ID3D12DescriptorHeap* heaps[] = {uav_desc_block.NativeDescriptorHeap()};
        d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);

        d3d12_cmd_list->ClearUnorderedAccessViewUint(uav_desc_block.GpuHandle(), uav.CpuHandle(), resource, color, 0, nullptr);

        gpu_system_->DeallocShaderVisibleCbvSrvUavDescBlock(std::move(uav_desc_block));
    }

    void GpuCommandList::ClearDepth(GpuDepthStencilView& dsv, float depth)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        dsv.Transition(*this);
        d3d12_cmd_list->ClearDepthStencilView(dsv.CpuHandle(), D3D12_CLEAR_FLAG_DEPTH, depth, 0, 0, nullptr);
    }

    void GpuCommandList::ClearStencil(GpuDepthStencilView& dsv, uint8_t stencil)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        dsv.Transition(*this);
        d3d12_cmd_list->ClearDepthStencilView(dsv.CpuHandle(), D3D12_CLEAR_FLAG_STENCIL, 0, stencil, 0, nullptr);
    }

    void GpuCommandList::ClearDepthStencil(GpuDepthStencilView& dsv, float depth, uint8_t stencil)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        dsv.Transition(*this);
        d3d12_cmd_list->ClearDepthStencilView(
            dsv.CpuHandle(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, depth, stencil, 0, nullptr);
    }

    void GpuCommandList::Render(const GpuRenderPipeline& pipeline, std::span<const VertexBufferBinding> vbs, const IndexBufferBinding* ib,
        uint32_t num, std::span<const ShaderBinding> shader_bindings, std::span<const GpuRenderTargetView*> rtvs,
        const GpuDepthStencilView* dsv, std::span<const GpuViewport> viewports, std::span<const GpuRect> scissor_rects)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        if (!vbs.empty())
        {
            auto vbvs = std::make_unique<D3D12_VERTEX_BUFFER_VIEW[]>(vbs.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(vbs.size()); ++i)
            {
                const auto& vb_binding = vbs[i];
                assert(vb_binding.vb != nullptr);

                vb_binding.vb->Transition(*this, GpuResourceState::Common);

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
            ib->ib->Transition(*this, GpuResourceState::Common);

            D3D12_INDEX_BUFFER_VIEW ibv;
            ibv.BufferLocation = ib->ib->GpuVirtualAddress() + ib->offset;
            ibv.SizeInBytes = ib->ib->Size();
            ibv.Format = ToDxgiFormat(ib->format);
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

        d3d12_cmd_list->IASetPrimitiveTopology(pipeline.NativePrimitiveTopology());

        d3d12_cmd_list->SetPipelineState(pipeline.NativePipelineState());
        d3d12_cmd_list->SetGraphicsRootSignature(pipeline.NativeRootSignature());

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
            heaps[num_heaps] = srv_uav_desc_block.NativeDescriptorHeap();
            ++num_heaps;
        }
        GpuDescriptorBlock sampler_desc_block;
        if (num_sampler_descs > 0)
        {
            sampler_desc_block = gpu_system_->AllocShaderVisibleSamplerDescBlock(num_sampler_descs);
            heaps[num_heaps] = sampler_desc_block.NativeDescriptorHeap();
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

            if (!binding.samplers.empty())
            {
                d3d12_cmd_list->SetGraphicsRootDescriptorTable(
                    root_index, OffsetHandle(sampler_desc_block.GpuHandle(), heap_base, sampler_desc_size));
                for (const auto* sampler : binding.samplers)
                {
                    if (sampler != nullptr)
                    {
                        auto sampler_cpu_handle = OffsetHandle(sampler_desc_block.CpuHandle(), heap_base, sampler_desc_size);
                        sampler->CopyTo(sampler_cpu_handle);
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
            D3D12_RECT d3d12_scissor_rect = {static_cast<LONG>(viewports[0].left), static_cast<LONG>(viewports[0].top),
                static_cast<LONG>(viewports[0].left + viewports[0].width), static_cast<LONG>(viewports[0].top + viewports[0].height)};
            d3d12_cmd_list->RSSetScissorRects(1, &d3d12_scissor_rect);
        }
        else
        {
            auto d3d12_scissor_rects = std::make_unique<D3D12_RECT[]>(scissor_rects.size());
            for (size_t i = 0; i < scissor_rects.size(); ++i)
            {
                d3d12_scissor_rects[i].left = scissor_rects[i].left;
                d3d12_scissor_rects[i].top = scissor_rects[i].top;
                d3d12_scissor_rects[i].right = scissor_rects[i].right;
                d3d12_scissor_rects[i].bottom = scissor_rects[i].bottom;
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

    void GpuCommandList::Compute(
        const GpuComputePipeline& pipeline, uint32_t group_x, uint32_t group_y, uint32_t group_z, const ShaderBinding& shader_binding)
    {
        assert(gpu_system_ != nullptr);

        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        d3d12_cmd_list->SetPipelineState(pipeline.NativePipelineState());
        d3d12_cmd_list->SetComputeRootSignature(pipeline.NativeRootSignature());

        const uint32_t num_srv_uav_descs = static_cast<uint32_t>(shader_binding.srvs.size() + shader_binding.uavs.size());
        const uint32_t num_sampler_descs = static_cast<uint32_t>(shader_binding.samplers.size());

        ID3D12DescriptorHeap* heaps[2] = {};
        uint32_t num_heaps = 0;

        GpuDescriptorBlock srv_uav_desc_block;
        if (num_srv_uav_descs > 0)
        {
            srv_uav_desc_block = gpu_system_->AllocShaderVisibleCbvSrvUavDescBlock(num_srv_uav_descs);
            heaps[num_heaps] = srv_uav_desc_block.NativeDescriptorHeap();
            ++num_heaps;
        }
        GpuDescriptorBlock sampler_desc_block;
        if (num_sampler_descs > 0)
        {
            sampler_desc_block = gpu_system_->AllocShaderVisibleSamplerDescBlock(num_sampler_descs);
            heaps[num_heaps] = sampler_desc_block.NativeDescriptorHeap();
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

        if (!shader_binding.samplers.empty())
        {
            d3d12_cmd_list->SetComputeRootDescriptorTable(
                root_index, OffsetHandle(sampler_desc_block.GpuHandle(), heap_base, sampler_desc_size));
            for (const auto* sampler : shader_binding.samplers)
            {
                if (sampler != nullptr)
                {
                    auto sampler_cpu_handle = OffsetHandle(sampler_desc_block.CpuHandle(), heap_base, sampler_desc_size);
                    sampler->CopyTo(sampler_cpu_handle);
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
        gpu_system_->DeallocShaderVisibleSamplerDescBlock(std::move(sampler_desc_block));
    }

    void GpuCommandList::ComputeIndirect(
        const GpuComputePipeline& pipeline, const GpuBuffer& indirect_args, const ShaderBinding& shader_binding)
    {
        assert(gpu_system_ != nullptr);

        indirect_args.Transition(*this, GpuResourceState::Common);

        GpuDescriptorBlock srv_uav_desc_block = this->BindPipeline(pipeline, shader_binding);

        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();
        d3d12_cmd_list->ExecuteIndirect(gpu_system_->NativeDispatchIndirectSignature(), 1, indirect_args.NativeBuffer(), 0, nullptr, 0);

        gpu_system_->DeallocShaderVisibleCbvSrvUavDescBlock(std::move(srv_uav_desc_block));
    }

    void GpuCommandList::Copy(GpuBuffer& dest, const GpuBuffer& src)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        src.Transition(*this, GpuResourceState::CopySrc);
        dest.Transition(*this, GpuResourceState::CopyDst);

        d3d12_cmd_list->CopyResource(dest.NativeBuffer(), src.NativeBuffer());
    }

    void GpuCommandList::Copy(GpuBuffer& dest, uint32_t dst_offset, const GpuBuffer& src, uint32_t src_offset, uint32_t src_size)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        src.Transition(*this, GpuResourceState::CopySrc);
        dest.Transition(*this, GpuResourceState::CopyDst);

        d3d12_cmd_list->CopyBufferRegion(dest.NativeBuffer(), dst_offset, src.NativeBuffer(), src_offset, src_size);
    }

    void GpuCommandList::Copy(GpuTexture& dest, const GpuTexture& src)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        src.Transition(*this, GpuResourceState::CopySrc);
        dest.Transition(*this, GpuResourceState::CopyDst);

        d3d12_cmd_list->CopyResource(dest.NativeTexture(), src.NativeTexture());
    }

    void GpuCommandList::Copy(GpuTexture& dest, uint32_t dest_sub_resource, uint32_t dst_x, uint32_t dst_y, uint32_t dst_z,
        const GpuTexture& src, uint32_t src_sub_resource, const GpuBox& src_box)
    {
        D3D12_TEXTURE_COPY_LOCATION src_loc;
        src_loc.pResource = src.NativeTexture();
        src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src_loc.SubresourceIndex = src_sub_resource;

        D3D12_TEXTURE_COPY_LOCATION dst_loc;
        dst_loc.pResource = dest.NativeTexture();
        dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst_loc.SubresourceIndex = dest_sub_resource;

        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        src.Transition(*this, GpuResourceState::CopySrc);
        dest.Transition(*this, GpuResourceState::CopyDst);

        const D3D12_BOX d3d12_src_box{src_box.left, src_box.top, src_box.front, src_box.right, src_box.bottom, src_box.back};
        d3d12_cmd_list->CopyTextureRegion(&dst_loc, dst_x, dst_y, dst_z, &src_loc, &d3d12_src_box);
    }

    void GpuCommandList::Upload(GpuBuffer& dest, const std::function<void(void*)>& copy_func)
    {
        D3D12_HEAP_PROPERTIES heap_prop;
        dest.NativeBuffer()->GetHeapProperties(&heap_prop, nullptr);

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

            auto upload_mem_block = gpu_system_->AllocUploadMemBlock(dest.Size(), GpuMemoryAllocator::StructuredDataAlignment);

            void* buff_data = upload_mem_block.CpuSpan<std::byte>().data();
            copy_func(buff_data);

            dest.Transition(*this, GpuResourceState::CopyDst);
            d3d12_cmd_list->CopyBufferRegion(
                dest.NativeBuffer(), 0, upload_mem_block.NativeBuffer(), upload_mem_block.Offset(), dest.Size());

            gpu_system_->DeallocUploadMemBlock(std::move(upload_mem_block));
        }
        break;

        default:
            Unreachable("Invalid heap type");
        }
    }

    void GpuCommandList::Upload(GpuBuffer& dest, const void* src_data, uint32_t src_size)
    {
        const uint32_t size = std::min(dest.Size(), src_size);
        this->Upload(dest, [src_data, size](void* dst_data) { std::memcpy(dst_data, src_data, size); });
    }

    void GpuCommandList::Upload(
        GpuTexture& dest, uint32_t sub_resource, const std::function<void(void*, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func)
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, dest.MipLevels(), dest.ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = dest.Width(mip);
        const uint32_t height = dest.Height(mip);
        const uint32_t depth = dest.Depth(mip);

        auto* d3d12_device = gpu_system_->NativeDevice();

        const auto desc = dest.NativeResource()->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
        uint64_t required_size = 0;
        d3d12_device->GetCopyableFootprints(&desc, sub_resource, 1, 0, &layout, nullptr, nullptr, &required_size);
        assert(layout.Footprint.RowPitch >= width * FormatSize(dest.Format()));

        auto upload_mem_block =
            gpu_system_->AllocUploadMemBlock(static_cast<uint32_t>(required_size), GpuMemoryAllocator::TextureDataAlignment);

        void* tex_data = upload_mem_block.CpuSpan<std::byte>().data();
        copy_func(tex_data, layout.Footprint.RowPitch, layout.Footprint.RowPitch * layout.Footprint.Height);

        layout.Offset += upload_mem_block.Offset();
        D3D12_TEXTURE_COPY_LOCATION src_loc;
        src_loc.pResource = upload_mem_block.NativeBuffer();
        src_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src_loc.PlacedFootprint = layout;

        D3D12_TEXTURE_COPY_LOCATION dst_loc;
        dst_loc.pResource = dest.NativeTexture();
        dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst_loc.SubresourceIndex = sub_resource;

        D3D12_BOX src_box;
        src_box.left = 0;
        src_box.top = 0;
        src_box.front = 0;
        src_box.right = width;
        src_box.bottom = height;
        src_box.back = depth;

        assert((type_ == GpuSystem::CmdQueueType::Render) || (type_ == GpuSystem::CmdQueueType::Compute));
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        dest.Transition(*this, GpuResourceState::CopyDst);

        d3d12_cmd_list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, &src_box);

        gpu_system_->DeallocUploadMemBlock(std::move(upload_mem_block));
    }

    void GpuCommandList::Upload(GpuTexture& dest, uint32_t sub_resource, const void* src_data, uint32_t src_size)
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, dest.MipLevels(), dest.ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = dest.Width(mip);
        const uint32_t height = dest.Height(mip);
        const uint32_t depth = dest.Depth(mip);
        const uint32_t src_row_pitch = width * FormatSize(dest.Format());

        this->Upload(dest, sub_resource,
            [height, depth, src_data, src_size, src_row_pitch](void* dst_data, uint32_t row_pitch, uint32_t slice_pitch) {
                uint32_t size = src_size;
                for (uint32_t z = 0; z < depth; ++z)
                {
                    for (uint32_t y = 0; y < height; ++y)
                    {
                        std::memcpy(&reinterpret_cast<std::byte*>(dst_data)[z * slice_pitch + y * row_pitch],
                            &reinterpret_cast<const std::byte*>(src_data)[(z * height + y) * src_row_pitch], std::min(src_row_pitch, size));
                        size -= src_row_pitch;
                    }
                }
            });
    }

    std::future<void> GpuCommandList::ReadBackAsync(const GpuBuffer& src, const std::function<void(const void*)>& copy_func)
    {
        D3D12_HEAP_PROPERTIES heap_prop;
        src.NativeBuffer()->GetHeapProperties(&heap_prop, nullptr);

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

            auto read_back_mem_block = gpu_system_->AllocReadBackMemBlock(src.Size(), GpuMemoryAllocator::StructuredDataAlignment);

            src.Transition(*this, GpuResourceState::CopySrc);

            d3d12_cmd_list->CopyBufferRegion(
                read_back_mem_block.NativeBuffer(), read_back_mem_block.Offset(), src.NativeBuffer(), 0, src.Size());
            const uint64_t fence_val = gpu_system_->ExecuteAndReset(*this);

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

    std::future<void> GpuCommandList::ReadBackAsync(const GpuBuffer& src, void* dst_data, uint32_t dst_size)
    {
        const uint32_t size = std::min(src.Size(), dst_size);
        return this->ReadBackAsync(src, [dst_data, size](const void* src_data) { std::memcpy(dst_data, src_data, size); });
    }

    std::future<void> GpuCommandList::ReadBackAsync(const GpuTexture& src, uint32_t sub_resource,
        const std::function<void(const void*, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func)
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, src.MipLevels(), src.ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = src.Width(mip);
        const uint32_t height = src.Height(mip);
        const uint32_t depth = src.Depth(mip);

        auto* d3d12_device = gpu_system_->NativeDevice();

        const auto desc = src.NativeResource()->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
        uint64_t required_size = 0;
        d3d12_device->GetCopyableFootprints(&desc, sub_resource, 1, 0, &layout, nullptr, nullptr, &required_size);
        assert(layout.Footprint.RowPitch >= width * FormatSize(src.Format()));

        auto read_back_mem_block =
            gpu_system_->AllocReadBackMemBlock(static_cast<uint32_t>(required_size), GpuMemoryAllocator::TextureDataAlignment);

        D3D12_TEXTURE_COPY_LOCATION src_loc;
        src_loc.pResource = src.NativeResource();
        src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src_loc.SubresourceIndex = sub_resource;

        layout.Offset = read_back_mem_block.Offset();
        D3D12_TEXTURE_COPY_LOCATION dst_loc;
        dst_loc.pResource = read_back_mem_block.NativeBuffer();
        dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst_loc.PlacedFootprint = layout;

        D3D12_BOX src_box;
        src_box.left = 0;
        src_box.top = 0;
        src_box.front = 0;
        src_box.right = width;
        src_box.bottom = height;
        src_box.back = depth;

        assert((type_ == GpuSystem::CmdQueueType::Render) || (type_ == GpuSystem::CmdQueueType::Compute));
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

        src.Transition(*this, GpuResourceState::CopySrc);

        d3d12_cmd_list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, &src_box);
        const uint64_t fence_val = gpu_system_->ExecuteAndReset(*this);

        return std::async(std::launch::deferred,
            [this, fence_val, read_back_mem_block = std::move(read_back_mem_block), row_pitch = layout.Footprint.RowPitch,
                slice_pitch = layout.Footprint.RowPitch * layout.Footprint.Height, copy_func]() mutable {
                gpu_system_->CpuWait(fence_val);

                const void* tex_data = read_back_mem_block.CpuSpan<std::byte>().data();
                copy_func(tex_data, row_pitch, slice_pitch);

                gpu_system_->DeallocReadBackMemBlock(std::move(read_back_mem_block));
            });
    }

    std::future<void> GpuCommandList::ReadBackAsync(const GpuTexture& src, uint32_t sub_resource, void* dst_data, uint32_t dst_size)
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, src.MipLevels(), src.ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = src.Width(mip);
        const uint32_t height = src.Height(mip);
        const uint32_t depth = src.Depth(mip);
        const uint32_t dst_row_pitch = width * FormatSize(src.Format());

        return this->ReadBackAsync(src, sub_resource,
            [height, depth, dst_data, dst_size, dst_row_pitch](const void* src_data, uint32_t row_pitch, uint32_t slice_pitch) {
                uint32_t size = dst_size;
                for (uint32_t z = 0; z < depth; ++z)
                {
                    for (uint32_t y = 0; y < height; ++y)
                    {
                        std::memcpy(&reinterpret_cast<std::byte*>(dst_data)[(z * height + y) * dst_row_pitch],
                            &reinterpret_cast<const std::byte*>(src_data)[z * slice_pitch + y * row_pitch], std::min(dst_row_pitch, size));
                        size -= dst_row_pitch;
                    }
                }
            });
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
        cmd_alloc_info_ = nullptr;
    }

    void GpuCommandList::Reset(GpuCommandAllocatorInfo& cmd_alloc_info)
    {
        cmd_alloc_info_ = &cmd_alloc_info;
        auto* d3d12_cmd_alloc = cmd_alloc_info.cmd_allocator.Get();
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
        closed_ = false;
    }

    GpuDescriptorBlock GpuCommandList::BindPipeline(const GpuComputePipeline& pipeline, const ShaderBinding& shader_binding)
    {
        auto* d3d12_cmd_list = this->NativeCommandList<ID3D12GraphicsCommandList>();

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

        return srv_uav_desc_block;
    }

    void GpuCommandList::GenerateMipmaps(GpuTexture2D& texture, GpuSampler::Filter filter)
    {
        auto& mipmapper = gpu_system_->Mipmapper();
        mipmapper.Generate(*this, texture, filter);
    }
} // namespace AIHoloImager
