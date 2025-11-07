// Copyright (c) 2025 Minmin Gong
//

#include "D3D12System.hpp"

#include <format>
#include <iostream>
#include <list>

#include <dxgi1_6.h>
#include <dxgidebug.h>

#include "Base/ErrorHandling.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Base/Uuid.hpp"
#include "Gpu/GpuCommandList.hpp"

#include "D3D12Buffer.hpp"
#include "D3D12CommandAllocatorInfo.hpp"
#include "D3D12CommandList.hpp"
#include "D3D12Conversion.hpp"
#include "D3D12DescriptorHeap.hpp"
#include "D3D12ImpDefine.hpp"
#include "D3D12ResourceViews.hpp"
#include "D3D12Sampler.hpp"
#include "D3D12Shader.hpp"
#include "D3D12Texture.hpp"
#include "D3D12VertexAttrib.hpp"

DEFINE_UUID_OF(IDXGIAdapter1);
DEFINE_UUID_OF(IDXGIFactory4);
DEFINE_UUID_OF(IDXGIFactory6);
DEFINE_UUID_OF(IDXGIInfoQueue);
DEFINE_UUID_OF(ID3D12CommandAllocator);
DEFINE_UUID_OF(ID3D12CommandQueue);
DEFINE_UUID_OF(ID3D12CommandSignature);
DEFINE_UUID_OF(ID3D12DescriptorHeap);
DEFINE_UUID_OF(ID3D12Debug);
DEFINE_UUID_OF(ID3D12Device);
DEFINE_UUID_OF(ID3D12Fence);
DEFINE_UUID_OF(ID3D12InfoQueue1);
DEFINE_UUID_OF(ID3D12PipelineState);
DEFINE_UUID_OF(ID3D12Resource);
DEFINE_UUID_OF(ID3D12RootSignature);

namespace AIHoloImager
{
    D3D12_IMP_IMP(System)

    D3D12System::D3D12System(
        GpuSystem& gpu_system, std::function<bool(void* device)> confirm_device, bool enable_sharing, bool enable_debug)
        : gpu_system_(&gpu_system)
    {
        bool debug_dxgi = false;
        ComPtr<IDXGIFactory4> dxgi_factory;
        if (enable_debug)
        {
            ComPtr<ID3D12Debug> debug_ctrl;
            if (SUCCEEDED(::D3D12GetDebugInterface(UuidOf<ID3D12Debug>(), debug_ctrl.PutVoid())))
            {
                debug_ctrl->EnableDebugLayer();
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
                    if (!confirm_device || confirm_device(device.Get()))
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

                d3d_info_queue->RegisterMessageCallback(
                    DebugMessageCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &dbg_callback_cookie_);
            }
        }

        TIFHR(device_->CreateFence(
            fence_val_, enable_sharing ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE, UuidOf<ID3D12Fence>(), fence_.PutVoid()));
        ++fence_val_;

        fence_event_ = MakeWin32UniqueHandle(::CreateEvent(nullptr, FALSE, FALSE, nullptr));
        Verify(fence_event_.get() != INVALID_HANDLE_VALUE);

        if (enable_sharing)
        {
            HANDLE shared_handle;
            TIFHR(device_->CreateSharedHandle(fence_.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle));
            shared_fence_handle_.reset(shared_handle);
        }

        {
            D3D12_INDIRECT_ARGUMENT_DESC indirect_param;
            indirect_param.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

            D3D12_COMMAND_SIGNATURE_DESC cmd_signature_desc;
            cmd_signature_desc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
            cmd_signature_desc.NumArgumentDescs = 1;
            cmd_signature_desc.pArgumentDescs = &indirect_param;
            cmd_signature_desc.NodeMask = 1;

            TIFHR(device_->CreateCommandSignature(
                &cmd_signature_desc, nullptr, UuidOf<ID3D12CommandSignature>(), dispatch_indirect_signature_.PutVoid()));
        }
    }

    D3D12System::~D3D12System()
    {
        this->CpuWait(GpuSystem::MaxFenceValue);

        stall_resources_.clear();

        fence_event_.reset();
        fence_ = nullptr;

        for (auto& cmd_queue : cmd_queues_)
        {
            cmd_queue.free_cmd_lists.clear();
            cmd_queue.cmd_allocator_infos.clear();
            cmd_queue.cmd_queue = nullptr;
        }

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

    ID3D12CommandQueue* D3D12System::CommandQueue(GpuSystem::CmdQueueType type) const noexcept
    {
        return cmd_queues_[static_cast<uint32_t>(type)].cmd_queue.Get();
    }

    void* D3D12System::NativeCommandQueue(GpuSystem::CmdQueueType type) const noexcept
    {
        return this->CommandQueue(type);
    }

    LUID D3D12System::DeviceLuid() const noexcept
    {
        return device_->GetAdapterLuid();
    }

    void* D3D12System::SharedFenceHandle() const noexcept
    {
        return shared_fence_handle_.get();
    }

    GpuCommandList D3D12System::CreateCommandList(GpuSystem::CmdQueueType type)
    {
        GpuCommandList cmd_list;
        auto& alloc_info = this->CurrentCommandAllocator(type);
        auto& cmd_queue = this->GetOrCreateCommandQueue(type);
        if (cmd_queue.free_cmd_lists.empty())
        {
            cmd_list = GpuCommandList(*gpu_system_, alloc_info, type);
        }
        else
        {
            cmd_list = std::move(cmd_queue.free_cmd_lists.front());
            cmd_queue.free_cmd_lists.pop_front();
            cmd_list.Reset(alloc_info);
        }
        return cmd_list;
    }

    uint64_t D3D12System::Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value)
    {
        const uint64_t new_fence_value = this->ExecuteOnly(D3D12Imp(cmd_list), wait_fence_value);
        this->GetOrCreateCommandQueue(cmd_list.Type()).free_cmd_lists.emplace_back(std::move(cmd_list));
        return new_fence_value;
    }

    uint64_t D3D12System::ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value)
    {
        return this->ExecuteAndReset(D3D12Imp(cmd_list), wait_fence_value);
    }

    uint64_t D3D12System::ExecuteAndReset(D3D12CommandList& cmd_list, uint64_t wait_fence_value)
    {
        const uint64_t new_fence_value = this->ExecuteOnly(cmd_list, wait_fence_value);
        cmd_list.Reset(this->CurrentCommandAllocator(cmd_list.Type()));
        return new_fence_value;
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

    void D3D12System::CpuWait(uint64_t fence_value)
    {
        if (fence_ && (fence_event_.get() != INVALID_HANDLE_VALUE))
        {
            for (auto& cmd_queue : cmd_queues_)
            {
                if (cmd_queue.cmd_queue)
                {
                    uint64_t wait_fence_value;
                    if (fence_value == GpuSystem::MaxFenceValue)
                    {
                        wait_fence_value = fence_val_;
                        if (FAILED(cmd_queue.cmd_queue->Signal(fence_.Get(), fence_val_)))
                        {
                            continue;
                        }
                        ++fence_val_;
                    }
                    else
                    {
                        wait_fence_value = fence_value;
                    }

                    if (fence_->GetCompletedValue() < wait_fence_value)
                    {
                        if (SUCCEEDED(fence_->SetEventOnCompletion(wait_fence_value, fence_event_.get())))
                        {
                            ::WaitForSingleObjectEx(fence_event_.get(), INFINITE, FALSE);
                        }
                    }
                }
            }
        }

        this->ClearStallResources();
    }

    void D3D12System::GpuWait(GpuSystem::CmdQueueType type, uint64_t fence_value)
    {
        if (fence_value != GpuSystem::MaxFenceValue)
        {
            fence_val_ = std::max(fence_val_, fence_value);
        }
        this->GetOrCreateCommandQueue(type).cmd_queue->Wait(fence_.Get(), fence_val_);
        ++fence_val_;
    }

    uint64_t D3D12System::FenceValue() const noexcept
    {
        return fence_val_;
    }

    uint64_t D3D12System::CompletedFenceValue() const
    {
        return fence_->GetCompletedValue();
    }

    void D3D12System::HandleDeviceLost()
    {
        for (auto& cmd_queue : cmd_queues_)
        {
            cmd_queue.cmd_queue.Reset();
            cmd_queue.cmd_allocator_infos.clear();
            cmd_queue.free_cmd_lists.clear();
        }

        fence_.Reset();
        device_.Reset();
    }

    void D3D12System::Recycle(ComPtr<ID3D12DeviceChild>&& resource)
    {
        stall_resources_.emplace_back(std::move(resource), fence_val_);
    }

    void D3D12System::ClearStallResources()
    {
        const uint64_t completed_fence = this->CompletedFenceValue();
        for (auto iter = stall_resources_.begin(); iter != stall_resources_.end();)
        {
            if (std::get<1>(*iter) <= completed_fence)
            {
                iter = stall_resources_.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }

    D3D12System::CmdQueue& D3D12System::GetOrCreateCommandQueue(GpuSystem::CmdQueueType type)
    {
        auto& cmd_queue = cmd_queues_[static_cast<uint32_t>(type)];
        if (!cmd_queue.cmd_queue)
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

            const D3D12_COMMAND_QUEUE_DESC queue_desc{d3d12_type, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
            TIFHR(device_->CreateCommandQueue(&queue_desc, UuidOf<ID3D12CommandQueue>(), cmd_queue.cmd_queue.PutVoid()));
            cmd_queue.cmd_queue->SetName(std::format(L"cmd_queue {}", static_cast<uint32_t>(type)).c_str());
        }

        return cmd_queue;
    }

    GpuCommandAllocatorInfo& D3D12System::CurrentCommandAllocator(GpuSystem::CmdQueueType type)
    {
        auto& cmd_queue = this->GetOrCreateCommandQueue(type);
        const uint64_t completed_fence = fence_->GetCompletedValue();
        for (auto& alloc : cmd_queue.cmd_allocator_infos)
        {
            auto& d3d12_alloc = D3D12Imp(*alloc);
            if (d3d12_alloc.EmptyAllocatedCommandLists() && (d3d12_alloc.FenceValue() <= completed_fence))
            {
                d3d12_alloc.CmdAllocator()->Reset();
                return *alloc;
            }
        }

        return *cmd_queue.cmd_allocator_infos.emplace_back(std::make_unique<GpuCommandAllocatorInfo>(*gpu_system_, type));
    }

    uint64_t D3D12System::ExecuteOnly(D3D12CommandList& cmd_list, uint64_t wait_fence_value)
    {
        auto& cmd_alloc_info = *cmd_list.CommandAllocatorInfo();
        cmd_list.Close();

        ID3D12CommandQueue* cmd_queue = this->GetOrCreateCommandQueue(cmd_list.Type()).cmd_queue.Get();

        if (wait_fence_value != GpuSystem::MaxFenceValue)
        {
            cmd_queue->Wait(fence_.Get(), wait_fence_value);
        }

        ID3D12CommandList* cmd_lists[] = {cmd_list.CommandListBase()};
        cmd_queue->ExecuteCommandLists(static_cast<uint32_t>(std::size(cmd_lists)), cmd_lists);

        const uint64_t curr_fence_value = fence_val_;
        TIFHR(cmd_queue->Signal(fence_.Get(), curr_fence_value));
        fence_val_ = curr_fence_value + 1;

        D3D12Imp(cmd_alloc_info).FenceValue(fence_val_);

        this->ClearStallResources();

        return curr_fence_value;
    }

    ID3D12CommandSignature* D3D12System::NativeDispatchIndirectSignature() const noexcept
    {
        return dispatch_indirect_signature_.Get();
    }

    std::unique_ptr<GpuBufferInternal> D3D12System::CreateBuffer(
        uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name) const
    {
        return std::make_unique<D3D12Buffer>(*gpu_system_, size, heap, flags, std::move(name));
    }
    std::unique_ptr<GpuBufferInternal> D3D12System::CreateBuffer(
        void* native_resource, GpuResourceState curr_state, std::wstring_view name) const
    {
        return std::make_unique<D3D12Buffer>(*gpu_system_, native_resource, curr_state, std::move(name));
    }

    std::unique_ptr<GpuTextureInternal> D3D12System::CreateTexture(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
        uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::wstring_view name) const
    {
        return std::make_unique<D3D12Texture>(
            *gpu_system_, type, width, height, depth, array_size, mip_levels, format, flags, std::move(name));
    }
    std::unique_ptr<GpuTextureInternal> D3D12System::CreateTexture(
        void* native_resource, GpuResourceState curr_state, std::wstring_view name) const
    {
        return std::make_unique<D3D12Texture>(*gpu_system_, native_resource, curr_state, std::move(name));
    }

    std::unique_ptr<GpuStaticSamplerInternal> D3D12System::CreateStaticSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<D3D12StaticSampler>(filters, addr_modes);
    }

    std::unique_ptr<GpuDynamicSamplerInternal> D3D12System::CreateDynamicSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<D3D12DynamicSampler>(*gpu_system_, filters, addr_modes);
    }

    std::unique_ptr<GpuVertexAttribsInternal> D3D12System::CreateVertexAttribs(std::span<const GpuVertexAttrib> attribs) const
    {
        return std::make_unique<D3D12VertexAttribs>(std::move(attribs));
    }

    std::unique_ptr<GpuDescriptorHeapInternal> D3D12System::CreateDescriptorHeap(
        uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name) const
    {
        return std::make_unique<D3D12DescriptorHeap>(*gpu_system_, size, type, shader_visible, std::move(name));
    }

    uint32_t D3D12System::DescriptorSize(GpuDescriptorHeapType type) const
    {
        return device_->GetDescriptorHandleIncrementSize(ToD3D12DescriptorHeapType(type));
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12System::CreateShaderResourceView(
        const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(*gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12System::CreateShaderResourceView(
        const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(*gpu_system_, texture_array, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12System::CreateShaderResourceView(
        const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(*gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12System::CreateShaderResourceView(
        const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(*gpu_system_, buffer, first_element, num_elements, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12System::CreateShaderResourceView(
        const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const
    {
        return std::make_unique<D3D12ShaderResourceView>(*gpu_system_, buffer, first_element, num_elements, element_size);
    }

    std::unique_ptr<GpuRenderTargetViewInternal> D3D12System::CreateRenderTargetView(GpuTexture2D& texture, GpuFormat format) const
    {
        return std::make_unique<D3D12RenderTargetView>(*gpu_system_, texture, format);
    }

    std::unique_ptr<GpuDepthStencilViewInternal> D3D12System::CreateDepthStencilView(GpuTexture2D& texture, GpuFormat format) const
    {
        return std::make_unique<D3D12DepthStencilView>(*gpu_system_, texture, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12System::CreateUnorderedAccessView(
        GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(*gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12System::CreateUnorderedAccessView(
        GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(*gpu_system_, texture_array, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12System::CreateUnorderedAccessView(
        GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(*gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12System::CreateUnorderedAccessView(
        GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(*gpu_system_, buffer, first_element, num_elements, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12System::CreateUnorderedAccessView(
        GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(*gpu_system_, buffer, first_element, num_elements, element_size);
    }

    std::unique_ptr<GpuRenderPipelineInternal> D3D12System::CreateRenderPipeline(GpuRenderPipeline::PrimitiveTopology topology,
        std::span<const ShaderInfo> shaders, const GpuVertexAttribs& vertex_attribs, std::span<const GpuStaticSampler> static_samplers,
        const GpuRenderPipeline::States& states) const
    {
        return std::make_unique<D3D12RenderPipeline>(
            *gpu_system_, topology, std::move(shaders), vertex_attribs, std::move(static_samplers), states);
    }

    std::unique_ptr<GpuComputePipelineInternal> D3D12System::CreateComputePipeline(
        const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers) const
    {
        return std::make_unique<D3D12ComputePipeline>(*gpu_system_, shader, std::move(static_samplers));
    }

    std::unique_ptr<GpuCommandAllocatorInfoInternal> D3D12System::CreateCommandAllocatorInfo(GpuSystem::CmdQueueType type) const
    {
        return std::make_unique<D3D12CommandAllocatorInfo>(*gpu_system_, type);
    }

    std::unique_ptr<GpuCommandListInternal> D3D12System::CreateCommandList(
        GpuCommandAllocatorInfo& cmd_alloc_info, GpuSystem::CmdQueueType type) const
    {
        return std::make_unique<D3D12CommandList>(*gpu_system_, cmd_alloc_info, type);
    }

    void D3D12System::DebugMessageCallback(D3D12_MESSAGE_CATEGORY category, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id,
        LPCSTR description, [[maybe_unused]] void* context)
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

        std::ostream& output_stream =
            (severity == D3D12_MESSAGE_SEVERITY_CORRUPTION || severity == D3D12_MESSAGE_SEVERITY_ERROR) ? std::cerr : std::cout;
        output_stream << std::format(
            "{}{}: {}({})[{}]: {}\n", color_escape, severity_str, EndEscape, category_str, static_cast<uint32_t>(id), description);
        output_stream.flush();
    }
} // namespace AIHoloImager
