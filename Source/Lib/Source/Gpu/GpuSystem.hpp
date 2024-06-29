// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <functional>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif
#include <directx/d3d12.h>

#include "GpuDescriptorAllocator.hpp"
#include "GpuMemoryAllocator.hpp"
#include "Util/ComPtr.hpp"
#include "Util/Noncopyable.hpp"
#include "Util/SmartPtrHelper.hpp"

namespace AIHoloImager
{
    class GpuSystem final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuSystem)

    public:
        static constexpr uint32_t FrameCount = 3;
        static constexpr uint64_t MaxFenceValue = ~0ull;

        enum class CmdQueueType : uint32_t
        {
            Render = 0,
            Compute,
            VideoEncode,

            Num,
        };

    public:
        explicit GpuSystem(std::function<bool(ID3D12Device* device)> confirm_device = nullptr);
        ~GpuSystem() noexcept;

        GpuSystem(GpuSystem&& other) noexcept;
        GpuSystem& operator=(GpuSystem&& other) noexcept;

        ID3D12Device* NativeDevice() const noexcept;
        ID3D12CommandQueue* NativeCommandQueue(CmdQueueType type) const noexcept;

        uint32_t FrameIndex() const noexcept;

        void MoveToNextFrame();

        [[nodiscard]] GpuCommandList CreateCommandList(CmdQueueType type);
        uint64_t Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value = MaxFenceValue);
        uint64_t ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value = MaxFenceValue);

        uint32_t RtvDescSize() const noexcept;
        uint32_t DsvDescSize() const noexcept;
        uint32_t CbvSrvUavDescSize() const noexcept;

        GpuDescriptorBlock AllocRtvDescBlock(uint32_t size);
        void DeallocRtvDescBlock(GpuDescriptorBlock&& desc_block);
        void ReallocRtvDescBlock(GpuDescriptorBlock& desc_block, uint32_t size);
        GpuDescriptorBlock AllocDsvDescBlock(uint32_t size);
        void DeallocDsvDescBlock(GpuDescriptorBlock&& desc_block);
        void ReallocDsvDescBlock(GpuDescriptorBlock& desc_block, uint32_t size);
        GpuDescriptorBlock AllocCbvSrvUavDescBlock(uint32_t size);
        void DeallocCbvSrvUavDescBlock(GpuDescriptorBlock&& desc_block);
        void ReallocCbvSrvUavDescBlock(GpuDescriptorBlock& desc_block, uint32_t size);

        GpuMemoryBlock AllocUploadMemBlock(uint32_t size_in_bytes, uint32_t alignment);
        void DeallocUploadMemBlock(GpuMemoryBlock&& mem_block);
        void ReallocUploadMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment);

        GpuMemoryBlock AllocReadbackMemBlock(uint32_t size_in_bytes, uint32_t alignment);
        void DeallocReadbackMemBlock(GpuMemoryBlock&& mem_block);
        void ReallocReadbackMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment);

        void WaitForGpu(uint64_t fence_value = MaxFenceValue);

        void HandleDeviceLost();

    private:
        ID3D12CommandAllocator* CurrentCommandAllocator(CmdQueueType type) const noexcept;
        uint64_t ExecuteOnly(GpuCommandList& cmd_list, uint64_t wait_fence_value);

    private:
        ComPtr<ID3D12Device> device_;

        struct CmdQueue
        {
            ComPtr<ID3D12CommandQueue> cmd_queue;
            ComPtr<ID3D12CommandAllocator> cmd_allocators[FrameCount];
            std::list<GpuCommandList> cmd_list_pool;
        };
        CmdQueue cmd_queues_[static_cast<uint32_t>(CmdQueueType::Num)];

        ComPtr<ID3D12Fence> fence_;
        uint64_t fence_vals_[FrameCount]{};
        Win32UniqueHandle fence_event_;

        uint32_t frame_index_ = 0;

        GpuMemoryAllocator upload_mem_allocator_;
        GpuMemoryAllocator readback_mem_allocator_;

        GpuDescriptorAllocator rtv_desc_allocator_;
        GpuDescriptorAllocator dsv_desc_allocator_;
        GpuDescriptorAllocator cbv_srv_uav_desc_allocator_;
    };
} // namespace AIHoloImager
