// Copyright (c) 2025-2026 Minmin Gong
//

#include "D3D12System.hpp"

#include <format>
#include <iostream>
#include <list>

#include <dxgi1_6.h>
#include <dxgidebug.h>

#include "Base/ErrorHandling.hpp"
#include "Base/Util.hpp"
#include "Base/Uuid.hpp"
#include "Gpu/GpuCommandList.hpp"

#include "D3D12Buffer.hpp"
#include "D3D12CommandList.hpp"
#include "D3D12CommandPool.hpp"
#include "D3D12CommandQueue.hpp"
#include "D3D12DescriptorHeap.hpp"
#include "D3D12Fence.hpp"
#include "D3D12ImpDefine.hpp"
#include "D3D12Query.hpp"
#include "D3D12ResourceViews.hpp"
#include "D3D12Sampler.hpp"
#include "D3D12Shader.hpp"
#include "D3D12Texture.hpp"
#include "D3D12VertexLayout.hpp"

DEFINE_UUID_OF(IDXGIAdapter1);
DEFINE_UUID_OF(IDXGIFactory4);
DEFINE_UUID_OF(IDXGIFactory6);
DEFINE_UUID_OF(IDXGIInfoQueue);
DEFINE_UUID_OF(ID3D12CommandSignature);
DEFINE_UUID_OF(ID3D12Debug);
DEFINE_UUID_OF(ID3D12Device);
DEFINE_UUID_OF(ID3D12DeviceRemovedExtendedData1);
DEFINE_UUID_OF(ID3D12DeviceRemovedExtendedDataSettings1);
DEFINE_UUID_OF(ID3D12InfoQueue1);
DEFINE_UUID_OF(ID3D12ShaderReflection);

const auto IID_IDxcUtils = __uuidof(IDxcUtils);
DEFINE_UUID_OF(IDxcUtils);

namespace
{
    std::string_view BreadcrumbToString(D3D12_AUTO_BREADCRUMB_OP op)
    {
        switch (static_cast<uint32_t>(op))
        {
        case D3D12_AUTO_BREADCRUMB_OP_SETMARKER:
            return "SetMarker";
        case D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT:
            return "BeginEvent";
        case D3D12_AUTO_BREADCRUMB_OP_ENDEVENT:
            return "EndEvent";
        case D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED:
            return "DrawInstanced";
        case D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED:
            return "DrawIndexedInstanced";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT:
            return "ExecuteIndirect";
        case D3D12_AUTO_BREADCRUMB_OP_DISPATCH:
            return "Dispatch";
        case D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION:
            return "CopyBufferRegion";
        case D3D12_AUTO_BREADCRUMB_OP_COPYTEXTUREREGION:
            return "CopyTextureRegion";
        case D3D12_AUTO_BREADCRUMB_OP_COPYRESOURCE:
            return "CopyResource";
        case D3D12_AUTO_BREADCRUMB_OP_COPYTILES:
            return "CopyTiles";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCE:
            return "ResolveSubresource";
        case D3D12_AUTO_BREADCRUMB_OP_CLEARRENDERTARGETVIEW:
            return "ClearRenderTargetView";
        case D3D12_AUTO_BREADCRUMB_OP_CLEARUNORDEREDACCESSVIEW:
            return "ClearUnorderedAccessView";
        case D3D12_AUTO_BREADCRUMB_OP_CLEARDEPTHSTENCILVIEW:
            return "ClearDepthStencilView";
        case D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER:
            return "ResourceBarrier";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEBUNDLE:
            return "ExecuteBundle";
        case D3D12_AUTO_BREADCRUMB_OP_PRESENT:
            return "Present";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVEQUERYDATA:
            return "ResolveQueryData";
        case D3D12_AUTO_BREADCRUMB_OP_BEGINSUBMISSION:
            return "BeginSubmission";
        case D3D12_AUTO_BREADCRUMB_OP_ENDSUBMISSION:
            return "EndSubmission";
        case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME:
            return "DecodeFrame";
        case D3D12_AUTO_BREADCRUMB_OP_PROCESSFRAMES:
            return "ProcessFrames";
        case D3D12_AUTO_BREADCRUMB_OP_ATOMICCOPYBUFFERUINT:
            return "AtomicCopyBufferUINT";
        case D3D12_AUTO_BREADCRUMB_OP_ATOMICCOPYBUFFERUINT64:
            return "AtomicCopyBufferUINT64";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCEREGION:
            return "ResolveSubresourceRegion";
        case D3D12_AUTO_BREADCRUMB_OP_WRITEBUFFERIMMEDIATE:
            return "WriteBufferImmediate";
        case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME1:
            return "DecodeFrame1";
        case D3D12_AUTO_BREADCRUMB_OP_SETPROTECTEDRESOURCESESSION:
            return "SetProtectedResourceSession";
        case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME2:
            return "DecodeFrame2";
        case D3D12_AUTO_BREADCRUMB_OP_PROCESSFRAMES1:
            return "ProcessFrames1";
        case D3D12_AUTO_BREADCRUMB_OP_BUILDRAYTRACINGACCELERATIONSTRUCTURE:
            return "BuildRaytracingAccelerationStructure";
        case D3D12_AUTO_BREADCRUMB_OP_EMITRAYTRACINGACCELERATIONSTRUCTUREPOSTBUILDINFO:
            return "EmitRaytracingAccelerationStructurePostbuildInfo";
        case D3D12_AUTO_BREADCRUMB_OP_COPYRAYTRACINGACCELERATIONSTRUCTURE:
            return "CopyRaytracingAccelerationStructure";
        case D3D12_AUTO_BREADCRUMB_OP_DISPATCHRAYS:
            return "DispatchRays";
        case D3D12_AUTO_BREADCRUMB_OP_INITIALIZEMETACOMMAND:
            return "InitializeMetaCommand";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEMETACOMMAND:
            return "ExecuteMetaCommand";
        case D3D12_AUTO_BREADCRUMB_OP_ESTIMATEMOTION:
            return "EstimateMotion";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVEMOTIONVECTORHEAP:
            return "ResolveMotionVectorHeap";
        case D3D12_AUTO_BREADCRUMB_OP_SETPIPELINESTATE1:
            return "SetPipelineState1";
        case D3D12_AUTO_BREADCRUMB_OP_INITIALIZEEXTENSIONCOMMAND:
            return "InitializeExtensionCommand";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEEXTENSIONCOMMAND:
            return "ExecuteExtensionCommand";
        case D3D12_AUTO_BREADCRUMB_OP_DISPATCHMESH:
            return "DispatchMesh";
        case D3D12_AUTO_BREADCRUMB_OP_ENCODEFRAME:
            return "EncodeFrame";
        case D3D12_AUTO_BREADCRUMB_OP_RESOLVEENCODEROUTPUTMETADATA:
            return "ResolveEncoderOutputMetadata";
        case 45: // D3D12_AUTO_BREADCRUMB_OP_BARRIER
            return "Barrier";
        case 46: // D3D12_AUTO_BREADCRUMB_OP_BEGIN_COMMAND_LIST
            return "BeginCommandList";

        default:
            return "";
        }
    }
} // namespace

namespace AIHoloImager
{
    D3D12_IMP_IMP(System)

    D3D12System::D3D12System(
        GpuSystem& gpu_system, std::function<bool(GpuSystem::Api api, void* device)> confirm_device, bool enable_sharing, bool enable_debug)
        : GpuSystemInternal(gpu_system, enable_sharing), rtv_desc_allocator_(gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, false),
          dsv_desc_allocator_(gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, false),
          cbv_srv_uav_desc_allocator_(gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, false),
          shader_visible_cbv_srv_uav_desc_allocator_(gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true),
          sampler_desc_allocator_(gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, false),
          shader_visible_sampler_desc_allocator_(gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, true)
    {
        bool debug_dxgi = false;
        ComPtr<IDXGIFactory4> dxgi_factory;
        if (enable_debug)
        {
            ComPtr<ID3D12Debug> debug_ctrl;
            if (SUCCEEDED(::D3D12GetDebugInterface(UuidOf<ID3D12Debug>(), debug_ctrl.PutVoid())))
            {
                debug_ctrl->EnableDebugLayer();

                ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> dred_settings;
                TIFHR(::D3D12GetDebugInterface(UuidOf<ID3D12DeviceRemovedExtendedDataSettings1>(), dred_settings.PutVoid()));
                dred_settings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
            else
            {
                ::OutputDebugStringW(L"WARNING: Direct3D Debug Device is not available\n");
            }

            ComPtr<IDXGIInfoQueue> dxgi_info_queue;
            if (SUCCEEDED(::DXGIGetDebugInterface1(0, UuidOf<IDXGIInfoQueue>(), dxgi_info_queue.PutVoid())))
            {
                debug_dxgi = true;

                TIFHR(::CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, UuidOf<IDXGIFactory4>(), dxgi_factory.PutVoid()));

                dxgi_info_queue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, true);
                dxgi_info_queue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, true);
            }
        }

        if (!debug_dxgi)
        {
            TIFHR(::CreateDXGIFactory2(0, UuidOf<IDXGIFactory4>(), dxgi_factory.PutVoid()));
        }

        {
            ComPtr<IDXGIFactory6> factory6 = dxgi_factory.As<IDXGIFactory6>();

            uint32_t adapter_id = 0;
            ComPtr<IDXGIAdapter1> adapter;
            while (factory6->EnumAdapterByGpuPreference(adapter_id, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, UuidOf<IDXGIAdapter1>(),
                       adapter.PutVoid()) != DXGI_ERROR_NOT_FOUND)
            {
                DXGI_ADAPTER_DESC1 desc;
                TIFHR(adapter->GetDesc1(&desc));

                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                {
                    continue;
                }

                ComPtr<ID3D12Device> device;
                if (SUCCEEDED(::D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, UuidOf<ID3D12Device>(), device.PutVoid())))
                {
                    if (!confirm_device || confirm_device(GpuSystem::Api::D3D12, device.Get()))
                    {
                        device_ = std::move(device);
                        break;
                    }
                }

                ++adapter_id;

                adapter.Reset();
            }
        }

        if (enable_debug && !device_)
        {
            ComPtr<IDXGIAdapter1> adapter;
            TIFHR(dxgi_factory->EnumWarpAdapter(UuidOf<IDXGIAdapter1>(), adapter.PutVoid()));

            TIFHR(::D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, UuidOf<ID3D12Device>(), device_.PutVoid()));
        }

        Verify(device_ != nullptr);

        if (enable_debug)
        {
            if (auto d3d_info_queue = device_.TryAs<ID3D12InfoQueue1>())
            {
                d3d_info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
                d3d_info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);

                D3D12_MESSAGE_ID deny_msg_ids[] = {
                    D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
                    D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
                };
                D3D12_INFO_QUEUE_FILTER filter{
                    .DenyList{
                        .NumIDs = std::size(deny_msg_ids),
                        .pIDList = deny_msg_ids,
                    },
                };
                d3d_info_queue->AddStorageFilterEntries(&filter);

                d3d_info_queue->RegisterMessageCallback(
                    DebugMessageCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, this, &dbg_callback_cookie_);
            }
        }

        {
            const D3D12_INDIRECT_ARGUMENT_DESC indirect_param{
                .Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
            };

            const D3D12_COMMAND_SIGNATURE_DESC cmd_signature_desc{
                .ByteStride = sizeof(D3D12_DRAW_ARGUMENTS),
                .NumArgumentDescs = 1,
                .pArgumentDescs = &indirect_param,
                .NodeMask = 1,
            };

            TIFHR(device_->CreateCommandSignature(
                &cmd_signature_desc, nullptr, UuidOf<ID3D12CommandSignature>(), draw_indirect_signature_.PutVoid()));
        }
        {
            const D3D12_INDIRECT_ARGUMENT_DESC indirect_param{
                .Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
            };

            const D3D12_COMMAND_SIGNATURE_DESC cmd_signature_desc{
                .ByteStride = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS),
                .NumArgumentDescs = 1,
                .pArgumentDescs = &indirect_param,
                .NodeMask = 1,
            };

            TIFHR(device_->CreateCommandSignature(
                &cmd_signature_desc, nullptr, UuidOf<ID3D12CommandSignature>(), draw_indexed_indirect_signature_.PutVoid()));
        }
        {
            const D3D12_INDIRECT_ARGUMENT_DESC indirect_param{
                .Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            };

            const D3D12_COMMAND_SIGNATURE_DESC cmd_signature_desc{
                .ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS),
                .NumArgumentDescs = 1,
                .pArgumentDescs = &indirect_param,
                .NodeMask = 1,
            };

            TIFHR(device_->CreateCommandSignature(
                &cmd_signature_desc, nullptr, UuidOf<ID3D12CommandSignature>(), dispatch_indirect_signature_.PutVoid()));
        }

        TIFHR(DxcCreateInstance(CLSID_DxcUtils, UuidOf<IDxcUtils>(), dxc_utils_.PutVoid()));
    }

    D3D12System::~D3D12System()
    {
        this->CpuWait(GpuSystem::WaitFences::Forever());

        rtv_desc_allocator_.Clear();
        shader_visible_sampler_desc_allocator_.Clear();
        sampler_desc_allocator_.Clear();
        shader_visible_cbv_srv_uav_desc_allocator_.Clear();
        cbv_srv_uav_desc_allocator_.Clear();
        dsv_desc_allocator_.Clear();

        stall_resources_.clear();

        this->ClearCommandQueueContexts();

        if (auto d3d_info_queue = device_.TryAs<ID3D12InfoQueue1>())
        {
            d3d_info_queue->UnregisterMessageCallback(dbg_callback_cookie_);
        }

        device_ = nullptr;
    }

    D3D12System::D3D12System(D3D12System&& other) noexcept = default;
    D3D12System::D3D12System(GpuSystemInternal&& other) noexcept : D3D12System(static_cast<D3D12System&&>(other))
    {
    }
    D3D12System& D3D12System::operator=(D3D12System&& other) noexcept = default;
    GpuSystemInternal& D3D12System::operator=(GpuSystemInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12System&&>(other));
    }

    ID3D12Device* D3D12System::Device() const noexcept
    {
        return device_.Get();
    }

    void* D3D12System::NativeDevice() const noexcept
    {
        return this->Device();
    }

    LUID D3D12System::DeviceLuid() const noexcept
    {
        return device_->GetAdapterLuid();
    }

    uint32_t D3D12System::ConstantDataAlignment() const noexcept
    {
        return D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    }
    uint32_t D3D12System::StructuredDataAlignment() const noexcept
    {
        return D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT;
    }
    uint32_t D3D12System::TextureDataAlignment() const noexcept
    {
        return D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    }

    void D3D12System::HandleDeviceLost()
    {
        rtv_desc_allocator_.Clear();
        dsv_desc_allocator_.Clear();
        cbv_srv_uav_desc_allocator_.Clear();
        shader_visible_cbv_srv_uav_desc_allocator_.Clear();
        sampler_desc_allocator_.Clear();
        shader_visible_sampler_desc_allocator_.Clear();

        this->ClearCommandQueueContexts();

        device_.Reset();
    }

    void D3D12System::Recycle(ComPtr<ID3D12DeviceChild>&& resource, std::shared_ptr<GpuSystem::WaitFences> wait_fences)
    {
        stall_resources_.emplace_back(std::move(resource), std::move(wait_fences));
    }

    void D3D12System::ClearStallResources()
    {
        GpuSystem::WaitFences wait_fences;
        for (uint32_t i = 0; i < std::size(wait_fences.fence_values); ++i)
        {
            const auto queue_type = static_cast<GpuSystem::CmdQueueType>(i);
            const uint64_t completed_fence = this->CompletedFenceValue(queue_type);
            wait_fences.fence_values[i] = completed_fence;
            if (completed_fence != 0)
            {
                for (auto res_iter = stall_resources_.begin(); res_iter != stall_resources_.end(); ++res_iter)
                {
                    if (res_iter->wait_fences->fence_values[i] <= completed_fence)
                    {
                        res_iter->wait_fences->fence_values[i] = 0;
                    }
                }
            }
        }

        rtv_desc_allocator_.ClearStallPages(wait_fences);
        dsv_desc_allocator_.ClearStallPages(wait_fences);
        cbv_srv_uav_desc_allocator_.ClearStallPages(wait_fences);
        shader_visible_cbv_srv_uav_desc_allocator_.ClearStallPages(wait_fences);
        sampler_desc_allocator_.ClearStallPages(wait_fences);
        shader_visible_sampler_desc_allocator_.ClearStallPages(wait_fences);

        for (auto iter = stall_resources_.begin(); iter != stall_resources_.end();)
        {
            bool all_completed = true;
            for (uint32_t i = 0; i < std::size(wait_fences.fence_values); ++i)
            {
                if (iter->wait_fences->fence_values[i] != 0)
                {
                    all_completed = false;
                    break;
                }
            }

            if (all_completed)
            {
                iter = stall_resources_.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }

    ID3D12CommandSignature* D3D12System::DrawIndirectSignature() const noexcept
    {
        return draw_indirect_signature_.Get();
    }
    ID3D12CommandSignature* D3D12System::DrawIndexedIndirectSignature() const noexcept
    {
        return draw_indexed_indirect_signature_.Get();
    }
    ID3D12CommandSignature* D3D12System::DispatchIndirectSignature() const noexcept
    {
        return dispatch_indirect_signature_.Get();
    }

    ComPtr<ID3D12ShaderReflection> D3D12System::ShaderReflect(std::span<const uint8_t> bytecode)
    {
        const DxcBuffer reflection_buffer{
            .Ptr = bytecode.data(),
            .Size = bytecode.size(),
            .Encoding = DXC_CP_ACP,
        };

        ComPtr<ID3D12ShaderReflection> reflection;
        TIFHR(dxc_utils_->CreateReflection(&reflection_buffer, UuidOf<ID3D12ShaderReflection>(), reflection.PutVoid()));
        return reflection;
    }

    uint32_t D3D12System::RtvDescSize() const noexcept
    {
        return rtv_desc_allocator_.DescriptorSize();
    }
    uint32_t D3D12System::DsvDescSize() const noexcept
    {
        return dsv_desc_allocator_.DescriptorSize();
    }
    uint32_t D3D12System::CbvSrvUavDescSize() const noexcept
    {
        return cbv_srv_uav_desc_allocator_.DescriptorSize();
    }
    uint32_t D3D12System::SamplerDescSize() const noexcept
    {
        return sampler_desc_allocator_.DescriptorSize();
    }

    std::unique_ptr<D3D12DescriptorHeap> D3D12System::CreateDescriptorHeap(
        uint32_t size, D3D12_DESCRIPTOR_HEAP_TYPE type, bool shader_visible, std::string_view name) const
    {
        return std::make_unique<D3D12DescriptorHeap>(this->GpuSys(), size, type, shader_visible, std::move(name));
    }

    uint32_t D3D12System::DescriptorSize(D3D12_DESCRIPTOR_HEAP_TYPE type) const
    {
        return device_->GetDescriptorHandleIncrementSize(type);
    }

    D3D12DescriptorBlock D3D12System::AllocRtvDescBlock(uint32_t size)
    {
        return rtv_desc_allocator_.Allocate(size);
    }
    void D3D12System::DeallocRtvDescBlock(D3D12DescriptorBlock&& desc_block)
    {
        return rtv_desc_allocator_.Deallocate(std::move(desc_block));
    }

    D3D12DescriptorBlock D3D12System::AllocDsvDescBlock(uint32_t size)
    {
        return dsv_desc_allocator_.Allocate(size);
    }
    void D3D12System::DeallocDsvDescBlock(D3D12DescriptorBlock&& desc_block)
    {
        return dsv_desc_allocator_.Deallocate(std::move(desc_block));
    }

    D3D12DescriptorBlock D3D12System::AllocCbvSrvUavDescBlock(uint32_t size)
    {
        return cbv_srv_uav_desc_allocator_.Allocate(size);
    }
    void D3D12System::DeallocCbvSrvUavDescBlock(D3D12DescriptorBlock&& desc_block)
    {
        return cbv_srv_uav_desc_allocator_.Deallocate(std::move(desc_block));
    }

    D3D12DescriptorBlock D3D12System::AllocShaderVisibleCbvSrvUavDescBlock(uint32_t size)
    {
        return shader_visible_cbv_srv_uav_desc_allocator_.Allocate(size);
    }
    void D3D12System::DeallocShaderVisibleCbvSrvUavDescBlock(D3D12DescriptorBlock&& desc_block)
    {
        return shader_visible_cbv_srv_uav_desc_allocator_.Deallocate(std::move(desc_block));
    }

    D3D12DescriptorBlock D3D12System::AllocSamplerDescBlock(uint32_t size)
    {
        return sampler_desc_allocator_.Allocate(size);
    }
    void D3D12System::DeallocSamplerDescBlock(D3D12DescriptorBlock&& desc_block)
    {
        return sampler_desc_allocator_.Deallocate(std::move(desc_block));
    }

    D3D12DescriptorBlock D3D12System::AllocShaderVisibleSamplerDescBlock(uint32_t size)
    {
        return shader_visible_sampler_desc_allocator_.Allocate(size);
    }
    void D3D12System::DeallocShaderVisibleSamplerDescBlock(D3D12DescriptorBlock&& desc_block)
    {
        return shader_visible_sampler_desc_allocator_.Deallocate(std::move(desc_block));
    }

    std::unique_ptr<GpuBufferInternal> D3D12System::CreateBuffer(
        uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name) const
    {
        return std::make_unique<D3D12Buffer>(this->GpuSys(), size, heap, flags, std::move(name));
    }

    std::unique_ptr<GpuTextureInternal> D3D12System::CreateTexture(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
        uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::string_view name) const
    {
        return std::make_unique<D3D12Texture>(
            this->GpuSys(), type, width, height, depth, array_size, mip_levels, format, flags, std::move(name));
    }

    std::unique_ptr<GpuStaticSamplerInternal> D3D12System::CreateStaticSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<D3D12StaticSampler>(filters, addr_modes);
    }

    std::unique_ptr<GpuDynamicSamplerInternal> D3D12System::CreateDynamicSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<D3D12DynamicSampler>(this->GpuSys(), filters, addr_modes);
    }

    std::unique_ptr<GpuVertexLayoutInternal> D3D12System::CreateVertexLayout(
        std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides) const
    {
        return std::make_unique<D3D12VertexLayout>(std::move(attribs), std::move(slot_strides));
    }

    std::unique_ptr<GpuConstantBufferViewInternal> D3D12System::CreateConstantBufferView(const GpuMemoryBlock& mem_block) const
    {
        return std::make_unique<D3D12ConstantBufferView>(mem_block);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12System::CreateShaderResourceView(
        const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(this->GpuSys(), texture, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12System::CreateShaderResourceView(
        const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(this->GpuSys(), texture_array, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12System::CreateShaderResourceView(
        const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(this->GpuSys(), texture, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12System::CreateShaderResourceView(
        const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(this->GpuSys(), buffer, first_element, num_elements, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12System::CreateShaderResourceView(
        const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const
    {
        return std::make_unique<D3D12ShaderResourceView>(this->GpuSys(), buffer, first_element, num_elements, element_size);
    }

    std::unique_ptr<GpuRenderTargetViewInternal> D3D12System::CreateRenderTargetView(GpuTexture2D& texture, GpuFormat format) const
    {
        return std::make_unique<D3D12RenderTargetView>(this->GpuSys(), texture, format);
    }

    std::unique_ptr<GpuDepthStencilViewInternal> D3D12System::CreateDepthStencilView(GpuTexture2D& texture, GpuFormat format) const
    {
        return std::make_unique<D3D12DepthStencilView>(this->GpuSys(), texture, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12System::CreateUnorderedAccessView(
        GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(this->GpuSys(), texture, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12System::CreateUnorderedAccessView(
        GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(this->GpuSys(), texture_array, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12System::CreateUnorderedAccessView(
        GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(this->GpuSys(), texture, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12System::CreateUnorderedAccessView(
        GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(this->GpuSys(), buffer, first_element, num_elements, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12System::CreateUnorderedAccessView(
        GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(this->GpuSys(), buffer, first_element, num_elements, element_size);
    }

    std::unique_ptr<GpuRenderPipelineInternal> D3D12System::CreateRenderPipeline(GpuRenderPipeline::PrimitiveTopology topology,
        std::span<const ShaderInfo> shaders, const GpuVertexLayout& vertex_layout, std::span<const GpuStaticSampler> static_samplers,
        const GpuRenderPipeline::States& states) const
    {
        return std::make_unique<D3D12RenderPipeline>(
            this->GpuSys(), topology, std::move(shaders), vertex_layout, std::move(static_samplers), states);
    }

    std::unique_ptr<GpuComputePipelineInternal> D3D12System::CreateComputePipeline(
        const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers) const
    {
        return std::make_unique<D3D12ComputePipeline>(this->GpuSys(), shader, std::move(static_samplers));
    }

    std::unique_ptr<GpuCommandPoolInternal> D3D12System::CreateCommandPool(GpuSystem::CmdQueueType type) const
    {
        return std::make_unique<D3D12CommandPool>(this->GpuSys(), type);
    }

    std::unique_ptr<GpuCommandListInternal> D3D12System::CreateCommandList(GpuCommandPool& cmd_pool, GpuSystem::CmdQueueType type) const
    {
        return std::make_unique<D3D12CommandList>(this->GpuSys(), cmd_pool, type);
    }

    std::unique_ptr<GpuTimerQueryInternal> D3D12System::CreateTimerQuery() const
    {
        return std::make_unique<D3D12TimerQuery>(this->GpuSys());
    }

    std::unique_ptr<GpuFenceInternal> D3D12System::CreateFence(uint64_t init_val, bool enable_sharing) const
    {
        return std::make_unique<D3D12Fence>(this->GpuSys(), init_val, enable_sharing);
    }

    std::unique_ptr<GpuCommandQueueInternal> D3D12System::CreateCommandQueue(GpuSystem::CmdQueueType type, std::string_view name) const
    {
        return std::make_unique<D3D12CommandQueue>(this->GpuSys(), type, std::move(name));
    }

    void D3D12System::DebugMessageCallback(
        D3D12_MESSAGE_CATEGORY category, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id, LPCSTR description, void* context)
    {
        constexpr const char* RedEscape = "\033[31m";
        constexpr const char* GreenEscape = "\033[32m";
        constexpr const char* YellowEscape = "\033[33m";
        constexpr const char* CyanEscape = "\033[36m";
        constexpr const char* EndEscape = "\033[0m";

        const char* color_escape;
        const char* severity_str;
        switch (severity)
        {
        case D3D12_MESSAGE_SEVERITY_CORRUPTION:
            color_escape = RedEscape;
            severity_str = "CORRUPTION";
            break;
        case D3D12_MESSAGE_SEVERITY_ERROR:
            color_escape = RedEscape;
            severity_str = "ERROR";
            break;
        case D3D12_MESSAGE_SEVERITY_WARNING:
            color_escape = YellowEscape;
            severity_str = "WARNING";
            break;
        case D3D12_MESSAGE_SEVERITY_INFO:
            color_escape = CyanEscape;
            severity_str = "INFO";
            break;
        case D3D12_MESSAGE_SEVERITY_MESSAGE:
            color_escape = GreenEscape;
            severity_str = "VERBOSE";
            break;

        default:
            Unreachable("Invalid D3D12 message severity");
        }

        const char* category_str;
        switch (category)
        {
        case D3D12_MESSAGE_CATEGORY_APPLICATION_DEFINED:
            category_str = "Application Defined";
            break;
        case D3D12_MESSAGE_CATEGORY_MISCELLANEOUS:
            category_str = "Miscellaneous";
            break;
        case D3D12_MESSAGE_CATEGORY_INITIALIZATION:
            category_str = "Initialization";
            break;
        case D3D12_MESSAGE_CATEGORY_CLEANUP:
            category_str = "Cleanup";
            break;
        case D3D12_MESSAGE_CATEGORY_COMPILATION:
            category_str = "Compilation";
            break;
        case D3D12_MESSAGE_CATEGORY_STATE_CREATION:
            category_str = "State Creation";
            break;
        case D3D12_MESSAGE_CATEGORY_STATE_SETTING:
            category_str = "State Setting";
            break;
        case D3D12_MESSAGE_CATEGORY_STATE_GETTING:
            category_str = "State Getting";
            break;
        case D3D12_MESSAGE_CATEGORY_RESOURCE_MANIPULATION:
            category_str = "Resource Manipulation";
            break;
        case D3D12_MESSAGE_CATEGORY_EXECUTION:
            category_str = "Execution";
            break;
        case D3D12_MESSAGE_CATEGORY_SHADER:
            category_str = "Shader";
            break;

        default:
            Unreachable("Invalid D3D12 message category");
        }

        const bool is_severe = (severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) || (severity == D3D12_MESSAGE_SEVERITY_ERROR);

        std::ostream& output_stream = is_severe ? std::cerr : std::cout;
        output_stream << std::format(
            "{}{}: {}({})[{}]: {}\n", color_escape, severity_str, EndEscape, category_str, static_cast<uint32_t>(id), description);
        output_stream.flush();

        if (is_severe)
        {
            auto* d3d12_device = static_cast<D3D12System*>(context)->device_.Get();
            const HRESULT removed_reason = d3d12_device->GetDeviceRemovedReason();
            if (removed_reason != S_OK)
            {
                output_stream << std::format(
                    "{}Device removed{}, reason code 0x{:X}\n", RedEscape, EndEscape, static_cast<uint32_t>(removed_reason));
                output_stream.flush();

                ComPtr<ID3D12DeviceRemovedExtendedData1> dred;
                TIFHR(d3d12_device->QueryInterface(UuidOf<ID3D12DeviceRemovedExtendedData1>(), dred.PutVoid()));

                D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 dred_auto_breadcrumbs_output;
                TIFHR(dred->GetAutoBreadcrumbsOutput1(&dred_auto_breadcrumbs_output));

                struct BreadcrumbNodeInfo
                {
                    std::string cmd_list_name;
                    std::string cmd_queue_name;

                    std::vector<D3D12_AUTO_BREADCRUMB_OP> ops;
                };

                std::vector<BreadcrumbNodeInfo> breadcrumbs_nodes;
                const D3D12_AUTO_BREADCRUMB_NODE1* head_auto_breadcrumb_node = dred_auto_breadcrumbs_output.pHeadAutoBreadcrumbNode;
                while (head_auto_breadcrumb_node != nullptr)
                {
                    BreadcrumbNodeInfo& info = breadcrumbs_nodes.emplace_back();
                    if (head_auto_breadcrumb_node->pCommandListDebugNameW != nullptr)
                    {
                        Convert(info.cmd_list_name,
                            std::u16string_view(reinterpret_cast<const char16_t*>(head_auto_breadcrumb_node->pCommandListDebugNameW)));
                    }
                    if (head_auto_breadcrumb_node->pCommandQueueDebugNameW != nullptr)
                    {
                        Convert(info.cmd_queue_name,
                            std::u16string_view(reinterpret_cast<const char16_t*>(head_auto_breadcrumb_node->pCommandQueueDebugNameW)));
                    }
                    info.ops.insert(info.ops.end(), head_auto_breadcrumb_node->pCommandHistory,
                        head_auto_breadcrumb_node->pCommandHistory + head_auto_breadcrumb_node->BreadcrumbCount);

                    head_auto_breadcrumb_node = head_auto_breadcrumb_node->pNext;
                }

                output_stream << std::format("{} breadcrumb nodes captured, from old to new:\n", breadcrumbs_nodes.size());
                for (const auto& node : breadcrumbs_nodes)
                {
                    output_stream << std::format("Breadcrumb - CmdList `{}` in CmdQueue `{}`\n",
                        node.cmd_list_name.empty() ? "<Unknown>" : node.cmd_list_name,
                        node.cmd_queue_name.empty() ? "<Unknown>" : node.cmd_queue_name);
                    for (const auto& op : node.ops)
                    {
                        output_stream << "    Op: ";
                        const std::string_view op_str = BreadcrumbToString(op);
                        if (op_str.empty())
                        {
                            output_stream << std::format("Unknown {}", static_cast<uint32_t>(op));
                        }
                        else
                        {
                            output_stream << op_str;
                        }
                        output_stream << '\n';
                    }
                }
                output_stream.flush();
            }
        }
    }
} // namespace AIHoloImager
