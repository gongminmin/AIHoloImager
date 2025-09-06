// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <functional>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Base/Noncopyable.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuDescriptorAllocator.hpp"
#include "Gpu/GpuMemoryAllocator.hpp"

namespace AIHoloImager
{
    struct GpuCommandAllocatorInfo
    {
        ComPtr<ID3D12CommandAllocator> cmd_allocator;
        uint64_t fence_val = 0;
    };

    class GpuSystem final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuSystem)

    public:
        static constexpr uint64_t MaxFenceValue = ~0ull;

        enum class CmdQueueType : uint32_t
        {
            Render = 0,
            Compute,
            VideoEncode,

            Num,
        };

    public:
        GpuSystem(
            std::function<bool(ID3D12Device* device)> confirm_device = nullptr, bool enable_sharing = false, bool enable_debug = false);
        ~GpuSystem();

        GpuSystem(GpuSystem&& other) noexcept;
        GpuSystem& operator=(GpuSystem&& other) noexcept;

        ID3D12Device* NativeDevice() const noexcept;
        ID3D12CommandQueue* NativeCommandQueue(CmdQueueType type) const noexcept;

        HANDLE SharedFenceHandle() const noexcept;

        [[nodiscard]] GpuCommandList CreateCommandList(CmdQueueType type);
        uint64_t Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value = MaxFenceValue);
        uint64_t ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value = MaxFenceValue);

        uint32_t RtvDescSize() const noexcept;
        uint32_t DsvDescSize() const noexcept;
        uint32_t CbvSrvUavDescSize() const noexcept;
        uint32_t SamplerDescSize() const noexcept;

        GpuDescriptorBlock AllocRtvDescBlock(uint32_t size);
        void DeallocRtvDescBlock(GpuDescriptorBlock&& desc_block);
        void ReallocRtvDescBlock(GpuDescriptorBlock& desc_block, uint32_t size);
        GpuDescriptorBlock AllocDsvDescBlock(uint32_t size);
        void DeallocDsvDescBlock(GpuDescriptorBlock&& desc_block);
        void ReallocDsvDescBlock(GpuDescriptorBlock& desc_block, uint32_t size);
        GpuDescriptorBlock AllocCbvSrvUavDescBlock(uint32_t size);
        void DeallocCbvSrvUavDescBlock(GpuDescriptorBlock&& desc_block);
        void ReallocCbvSrvUavDescBlock(GpuDescriptorBlock& desc_block, uint32_t size);
        GpuDescriptorBlock AllocShaderVisibleCbvSrvUavDescBlock(uint32_t size);
        void DeallocShaderVisibleCbvSrvUavDescBlock(GpuDescriptorBlock&& desc_block);
        void ReallocShaderVisibleCbvSrvUavDescBlock(GpuDescriptorBlock& desc_block, uint32_t size);
        GpuDescriptorBlock AllocSamplerDescBlock(uint32_t size);
        void DeallocSamplerDescBlock(GpuDescriptorBlock&& desc_block);
        void ReallocSamplerDescBlock(GpuDescriptorBlock& desc_block, uint32_t size);
        GpuDescriptorBlock AllocShaderVisibleSamplerDescBlock(uint32_t size);
        void DeallocShaderVisibleSamplerDescBlock(GpuDescriptorBlock&& desc_block);
        void ReallocShaderVisibleSamplerDescBlock(GpuDescriptorBlock& desc_block, uint32_t size);

        GpuMemoryBlock AllocUploadMemBlock(uint32_t size_in_bytes, uint32_t alignment);
        void DeallocUploadMemBlock(GpuMemoryBlock&& mem_block);
        void ReallocUploadMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment);

        GpuMemoryBlock AllocReadBackMemBlock(uint32_t size_in_bytes, uint32_t alignment);
        void DeallocReadBackMemBlock(GpuMemoryBlock&& mem_block);
        void ReallocReadBackMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment);

        void CpuWait(uint64_t fence_value = MaxFenceValue);
        void GpuWait(CmdQueueType type, uint64_t fence_value = MaxFenceValue);
        uint64_t FenceValue() const noexcept;

        void HandleDeviceLost();

        void Recycle(ComPtr<ID3D12DeviceChild>&& resource);

        ID3D12CommandSignature* NativeDispatchIndirectSignature() const noexcept;

    private:
        struct CmdQueue
        {
            ComPtr<ID3D12CommandQueue> cmd_queue;
            std::vector<std::unique_ptr<GpuCommandAllocatorInfo>> cmd_allocator_infos;
            std::list<GpuCommandList> free_cmd_lists;
        };

    private:
        CmdQueue& GetOrCreateCommandQueue(CmdQueueType type);
        GpuCommandAllocatorInfo& CurrentCommandAllocator(CmdQueueType type);
        uint64_t ExecuteOnly(GpuCommandList& cmd_list, uint64_t wait_fence_value);
        void ClearStallResources();

    private:
        ComPtr<ID3D12Device> device_;

        CmdQueue cmd_queues_[static_cast<uint32_t>(CmdQueueType::Num)];

        ComPtr<ID3D12Fence> fence_;
        uint64_t fence_val_ = 0;
        Win32UniqueHandle fence_event_;
        Win32UniqueHandle shared_fence_handle_;

        GpuMemoryAllocator upload_mem_allocator_;
        GpuMemoryAllocator read_back_mem_allocator_;

        GpuDescriptorAllocator rtv_desc_allocator_;
        GpuDescriptorAllocator dsv_desc_allocator_;
        GpuDescriptorAllocator cbv_srv_uav_desc_allocator_;
        GpuDescriptorAllocator shader_visible_cbv_srv_uav_desc_allocator_;
        GpuDescriptorAllocator sampler_desc_allocator_;
        GpuDescriptorAllocator shader_visible_sampler_desc_allocator_;

        std::list<std::tuple<ComPtr<ID3D12DeviceChild>, uint64_t>> stall_resources_;

        ComPtr<ID3D12CommandSignature> dispatch_indirect_signature_;
    };

    constexpr uint32_t DivUp(uint32_t a, uint32_t b) noexcept
    {
        return (a + b - 1) / b;
    }
} // namespace AIHoloImager
