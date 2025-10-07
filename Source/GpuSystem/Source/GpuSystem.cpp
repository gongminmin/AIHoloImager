// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuSystem.hpp"

#include <format>
#include <list>

#include <dxgi1_6.h>
#include <dxgidebug.h>

#include "Base/ErrorHandling.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Base/Uuid.hpp"
#include "Gpu/GpuCommandList.hpp"

#include "Internal/D3D12/D3D12SystemFactory.hpp"
#include "Internal/GpuSystemInternalFactory.hpp"

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
DEFINE_UUID_OF(ID3D12InfoQueue);
DEFINE_UUID_OF(ID3D12PipelineState);
DEFINE_UUID_OF(ID3D12Resource);
DEFINE_UUID_OF(ID3D12RootSignature);

namespace AIHoloImager
{
    GpuSystem::GpuSystem(std::function<bool(ID3D12Device* device)> confirm_device, bool enable_sharing, bool enable_debug)
        : internal_factory_(std::make_unique<D3D12SystemFactory>(*this)), upload_mem_allocator_(*this, true),
          read_back_mem_allocator_(*this, false), rtv_desc_allocator_(*this, GpuDescriptorHeapType::Rtv, D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
          dsv_desc_allocator_(*this, GpuDescriptorHeapType::Dsv, D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
          cbv_srv_uav_desc_allocator_(*this, GpuDescriptorHeapType::CbvSrvUav, D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
          shader_visible_cbv_srv_uav_desc_allocator_(*this, GpuDescriptorHeapType::CbvSrvUav, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
          sampler_desc_allocator_(*this, GpuDescriptorHeapType::Sampler, D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
          shader_visible_sampler_desc_allocator_(*this, GpuDescriptorHeapType::Sampler, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
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
            if (ComPtr<ID3D12InfoQueue> d3d_info_queue = device_.TryAs<ID3D12InfoQueue>())
            {
                d3d_info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
                d3d_info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
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

        mipmapper_ = GpuMipmapper(*this);
    }

    GpuSystem::~GpuSystem()
    {
        this->CpuWait();

        shader_visible_sampler_desc_allocator_.Clear();
        sampler_desc_allocator_.Clear();
        shader_visible_cbv_srv_uav_desc_allocator_.Clear();
        cbv_srv_uav_desc_allocator_.Clear();
        dsv_desc_allocator_.Clear();
        rtv_desc_allocator_.Clear();

        read_back_mem_allocator_.Clear();
        upload_mem_allocator_.Clear();

        stall_resources_.clear();

        fence_event_.reset();
        fence_ = nullptr;

        for (auto& cmd_queue : cmd_queues_)
        {
            cmd_queue.free_cmd_lists.clear();
            cmd_queue.cmd_allocator_infos.clear();
            cmd_queue.cmd_queue = nullptr;
        }

        device_ = nullptr;
    }

    GpuSystem::GpuSystem(GpuSystem&& other) noexcept = default;
    GpuSystem& GpuSystem::operator=(GpuSystem&& other) noexcept = default;

    ID3D12Device* GpuSystem::NativeDevice() const noexcept
    {
        return device_.Get();
    }

    ID3D12CommandQueue* GpuSystem::NativeCommandQueue(CmdQueueType type) const noexcept
    {
        return cmd_queues_[static_cast<uint32_t>(type)].cmd_queue.Get();
    }

    HANDLE GpuSystem::SharedFenceHandle() const noexcept
    {
        return shared_fence_handle_.get();
    }

    GpuCommandList GpuSystem::CreateCommandList(GpuSystem::CmdQueueType type)
    {
        GpuCommandList cmd_list;
        auto& alloc_info = this->CurrentCommandAllocator(type);
        auto& cmd_queue = this->GetOrCreateCommandQueue(type);
        if (cmd_queue.free_cmd_lists.empty())
        {
            cmd_list = GpuCommandList(*this, alloc_info, type);
        }
        else
        {
            cmd_list = std::move(cmd_queue.free_cmd_lists.front());
            cmd_queue.free_cmd_lists.pop_front();
            cmd_list.Reset(alloc_info);
        }
        return cmd_list;
    }

    uint64_t GpuSystem::Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value)
    {
        const uint64_t new_fence_value = this->ExecuteOnly(cmd_list, wait_fence_value);
        this->GetOrCreateCommandQueue(cmd_list.Type()).free_cmd_lists.emplace_back(std::move(cmd_list));
        return new_fence_value;
    }

    uint64_t GpuSystem::ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value)
    {
        const uint64_t new_fence_value = this->ExecuteOnly(cmd_list, wait_fence_value);
        cmd_list.Reset(this->CurrentCommandAllocator(cmd_list.Type()));
        return new_fence_value;
    }

    uint32_t GpuSystem::RtvDescSize() const noexcept
    {
        return rtv_desc_allocator_.DescriptorSize();
    }

    uint32_t GpuSystem::DsvDescSize() const noexcept
    {
        return dsv_desc_allocator_.DescriptorSize();
    }

    uint32_t GpuSystem::CbvSrvUavDescSize() const noexcept
    {
        return cbv_srv_uav_desc_allocator_.DescriptorSize();
    }

    uint32_t GpuSystem::SamplerDescSize() const noexcept
    {
        return sampler_desc_allocator_.DescriptorSize();
    }

    GpuDescriptorBlock GpuSystem::AllocRtvDescBlock(uint32_t size)
    {
        return rtv_desc_allocator_.Allocate(size);
    }

    void GpuSystem::DeallocRtvDescBlock(GpuDescriptorBlock&& desc_block)
    {
        return rtv_desc_allocator_.Deallocate(std::move(desc_block), fence_val_);
    }

    void GpuSystem::ReallocRtvDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        return rtv_desc_allocator_.Reallocate(desc_block, fence_val_, size);
    }

    GpuDescriptorBlock GpuSystem::AllocDsvDescBlock(uint32_t size)
    {
        return dsv_desc_allocator_.Allocate(size);
    }

    void GpuSystem::DeallocDsvDescBlock(GpuDescriptorBlock&& desc_block)
    {
        return dsv_desc_allocator_.Deallocate(std::move(desc_block), fence_val_);
    }

    void GpuSystem::ReallocDsvDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        return dsv_desc_allocator_.Reallocate(desc_block, fence_val_, size);
    }

    GpuDescriptorBlock GpuSystem::AllocCbvSrvUavDescBlock(uint32_t size)
    {
        return cbv_srv_uav_desc_allocator_.Allocate(size);
    }

    void GpuSystem::DeallocCbvSrvUavDescBlock(GpuDescriptorBlock&& desc_block)
    {
        return cbv_srv_uav_desc_allocator_.Deallocate(std::move(desc_block), fence_val_);
    }

    void GpuSystem::ReallocCbvSrvUavDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        return cbv_srv_uav_desc_allocator_.Reallocate(desc_block, fence_val_, size);
    }

    GpuDescriptorBlock GpuSystem::AllocShaderVisibleCbvSrvUavDescBlock(uint32_t size)
    {
        return shader_visible_cbv_srv_uav_desc_allocator_.Allocate(size);
    }

    void GpuSystem::DeallocShaderVisibleCbvSrvUavDescBlock(GpuDescriptorBlock&& desc_block)
    {
        return shader_visible_cbv_srv_uav_desc_allocator_.Deallocate(std::move(desc_block), fence_val_);
    }

    void GpuSystem::ReallocShaderVisibleCbvSrvUavDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        return shader_visible_cbv_srv_uav_desc_allocator_.Reallocate(desc_block, fence_val_, size);
    }

    GpuDescriptorBlock GpuSystem::AllocSamplerDescBlock(uint32_t size)
    {
        return sampler_desc_allocator_.Allocate(size);
    }

    void GpuSystem::DeallocSamplerDescBlock(GpuDescriptorBlock&& desc_block)
    {
        return sampler_desc_allocator_.Deallocate(std::move(desc_block), fence_val_);
    }

    void GpuSystem::ReallocSamplerDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        return sampler_desc_allocator_.Reallocate(desc_block, fence_val_, size);
    }

    GpuDescriptorBlock GpuSystem::AllocShaderVisibleSamplerDescBlock(uint32_t size)
    {
        return shader_visible_sampler_desc_allocator_.Allocate(size);
    }

    void GpuSystem::DeallocShaderVisibleSamplerDescBlock(GpuDescriptorBlock&& desc_block)
    {
        return shader_visible_sampler_desc_allocator_.Deallocate(std::move(desc_block), fence_val_);
    }

    void GpuSystem::ReallocShaderVisibleSamplerDescBlock(GpuDescriptorBlock& desc_block, uint32_t size)
    {
        return shader_visible_sampler_desc_allocator_.Reallocate(desc_block, fence_val_, size);
    }

    GpuMemoryBlock GpuSystem::AllocUploadMemBlock(uint32_t size_in_bytes, uint32_t alignment)
    {
        return upload_mem_allocator_.Allocate(size_in_bytes, alignment);
    }

    void GpuSystem::DeallocUploadMemBlock(GpuMemoryBlock&& mem_block)
    {
        return upload_mem_allocator_.Deallocate(std::move(mem_block), fence_val_);
    }

    void GpuSystem::ReallocUploadMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment)
    {
        return upload_mem_allocator_.Reallocate(mem_block, fence_val_, size_in_bytes, alignment);
    }

    GpuMemoryBlock GpuSystem::AllocReadBackMemBlock(uint32_t size_in_bytes, uint32_t alignment)
    {
        return read_back_mem_allocator_.Allocate(size_in_bytes, alignment);
    }

    void GpuSystem::DeallocReadBackMemBlock(GpuMemoryBlock&& mem_block)
    {
        return read_back_mem_allocator_.Deallocate(std::move(mem_block), fence_val_);
    }

    void GpuSystem::ReallocReadBackMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment)
    {
        return read_back_mem_allocator_.Reallocate(mem_block, fence_val_, size_in_bytes, alignment);
    }

    void GpuSystem::CpuWait(uint64_t fence_value)
    {
        if (fence_ && (fence_event_.get() != INVALID_HANDLE_VALUE))
        {
            for (auto& cmd_queue : cmd_queues_)
            {
                if (cmd_queue.cmd_queue)
                {
                    uint64_t wait_fence_value;
                    if (fence_value == MaxFenceValue)
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

    void GpuSystem::GpuWait(CmdQueueType type, uint64_t fence_value)
    {
        if (fence_value != MaxFenceValue)
        {
            fence_val_ = std::max(fence_val_, fence_value);
        }
        this->GetOrCreateCommandQueue(type).cmd_queue->Wait(fence_.Get(), fence_val_);
        ++fence_val_;
    }

    uint64_t GpuSystem::FenceValue() const noexcept
    {
        return fence_val_;
    }

    void GpuSystem::HandleDeviceLost()
    {
        upload_mem_allocator_.Clear();
        read_back_mem_allocator_.Clear();

        rtv_desc_allocator_.Clear();
        dsv_desc_allocator_.Clear();
        cbv_srv_uav_desc_allocator_.Clear();
        shader_visible_cbv_srv_uav_desc_allocator_.Clear();
        sampler_desc_allocator_.Clear();
        shader_visible_sampler_desc_allocator_.Clear();

        for (auto& cmd_queue : cmd_queues_)
        {
            cmd_queue.cmd_queue.Reset();
            cmd_queue.cmd_allocator_infos.clear();
            cmd_queue.free_cmd_lists.clear();
        }

        fence_.Reset();
        device_.Reset();
    }

    void GpuSystem::Recycle(ComPtr<ID3D12DeviceChild>&& resource)
    {
        stall_resources_.emplace_back(std::move(resource), fence_val_);
    }

    void GpuSystem::ClearStallResources()
    {
        const uint64_t completed_fence = fence_->GetCompletedValue();
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

        upload_mem_allocator_.ClearStallPages(completed_fence);
        read_back_mem_allocator_.ClearStallPages(completed_fence);

        rtv_desc_allocator_.ClearStallPages(completed_fence);
        dsv_desc_allocator_.ClearStallPages(completed_fence);
        cbv_srv_uav_desc_allocator_.ClearStallPages(completed_fence);
        shader_visible_cbv_srv_uav_desc_allocator_.ClearStallPages(completed_fence);
        sampler_desc_allocator_.ClearStallPages(completed_fence);
        shader_visible_sampler_desc_allocator_.ClearStallPages(completed_fence);
    }

    GpuSystem::CmdQueue& GpuSystem::GetOrCreateCommandQueue(CmdQueueType type)
    {
        auto& cmd_queue = cmd_queues_[static_cast<uint32_t>(type)];
        if (!cmd_queue.cmd_queue)
        {
            D3D12_COMMAND_LIST_TYPE d3d12_type;
            switch (type)
            {
            case CmdQueueType::Render:
                d3d12_type = D3D12_COMMAND_LIST_TYPE_DIRECT;
                break;

            case CmdQueueType::Compute:
                d3d12_type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
                break;

            case CmdQueueType::VideoEncode:
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

    GpuCommandAllocatorInfo& GpuSystem::CurrentCommandAllocator(GpuSystem::CmdQueueType type)
    {
        auto& cmd_queue = this->GetOrCreateCommandQueue(type);
        const uint64_t completed_fence = fence_->GetCompletedValue();
        for (auto& alloc : cmd_queue.cmd_allocator_infos)
        {
            if (alloc->fence_val <= completed_fence)
            {
                alloc->cmd_allocator->Reset();
                return *alloc;
            }
        }

        D3D12_COMMAND_LIST_TYPE d3d12_type;
        switch (type)
        {
        case CmdQueueType::Render:
            d3d12_type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            break;

        case CmdQueueType::Compute:
            d3d12_type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            break;

        case CmdQueueType::VideoEncode:
            d3d12_type = D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE;
            break;

        default:
            Unreachable();
        }

        auto& alloc = *cmd_queue.cmd_allocator_infos.emplace_back(std::make_unique<GpuCommandAllocatorInfo>());
        TIFHR(device_->CreateCommandAllocator(d3d12_type, UuidOf<ID3D12CommandAllocator>(), alloc.cmd_allocator.PutVoid()));
        return alloc;
    }

    uint64_t GpuSystem::ExecuteOnly(GpuCommandList& cmd_list, uint64_t wait_fence_value)
    {
        auto& cmd_alloc_info = *cmd_list.CommandAllocatorInfo();
        cmd_list.Close();

        ID3D12CommandQueue* cmd_queue = this->GetOrCreateCommandQueue(cmd_list.Type()).cmd_queue.Get();

        if (wait_fence_value != MaxFenceValue)
        {
            cmd_queue->Wait(fence_.Get(), wait_fence_value);
        }

        ID3D12CommandList* cmd_lists[] = {cmd_list.NativeCommandListBase()};
        cmd_queue->ExecuteCommandLists(static_cast<uint32_t>(std::size(cmd_lists)), cmd_lists);

        const uint64_t curr_fence_value = fence_val_;
        TIFHR(cmd_queue->Signal(fence_.Get(), curr_fence_value));
        fence_val_ = curr_fence_value + 1;

        cmd_alloc_info.fence_val = fence_val_;

        this->ClearStallResources();

        return curr_fence_value;
    }

    ID3D12CommandSignature* GpuSystem::NativeDispatchIndirectSignature() const noexcept
    {
        return dispatch_indirect_signature_.Get();
    }

    GpuMipmapper& GpuSystem::Mipmapper() noexcept
    {
        return mipmapper_;
    }

    const GpuSystemInternalFactory& GpuSystem::InternalFactory() const noexcept
    {
        return *internal_factory_;
    }
} // namespace AIHoloImager
