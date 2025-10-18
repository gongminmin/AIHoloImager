// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <functional>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuSystem.hpp"

#include "../GpuCommandListInternal.hpp"
#include "../GpuSystemInternal.hpp"

namespace AIHoloImager
{
    class D3D12System : public GpuSystemInternal
    {
    public:
        D3D12System(GpuSystem& gpu_system_, std::function<bool(void* device)> confirm_device = nullptr, bool enable_sharing = false,
            bool enable_debug = false);
        ~D3D12System() override;

        D3D12System(D3D12System&& other) noexcept;
        explicit D3D12System(GpuSystemInternal&& other) noexcept;
        D3D12System& operator=(D3D12System&& other) noexcept;
        GpuSystemInternal& operator=(GpuSystemInternal&& other) noexcept override;

        void* NativeDevice() const noexcept override;
        template <typename Traits>
        typename Traits::DeviceType NativeDevice() const noexcept
        {
            return reinterpret_cast<typename Traits::DeviceType>(this->NativeDevice());
        }
        void* NativeCommandQueue(GpuSystem::CmdQueueType type) const noexcept override;
        template <typename Traits>
        typename Traits::CommandQueueType NativeCommandQueue() const noexcept
        {
            return reinterpret_cast<typename Traits::CommandQueueType>(this->NativeCommandQueue());
        }

        void* SharedFenceHandle() const noexcept override;

        [[nodiscard]] GpuCommandList CreateCommandList(GpuSystem::CmdQueueType type) override;
        uint64_t Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value) override;
        uint64_t ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value) override;
        uint64_t ExecuteAndReset(GpuCommandListInternal& cmd_list, uint64_t wait_fence_value);

        uint32_t ConstantDataAlignment() const noexcept override;
        uint32_t StructuredDataAlignment() const noexcept override;
        uint32_t TextureDataAlignment() const noexcept override;

        void CpuWait(uint64_t fence_value) override;
        void GpuWait(GpuSystem::CmdQueueType type, uint64_t fence_value) override;
        uint64_t FenceValue() const noexcept override;
        uint64_t CompletedFenceValue() const override;

        void HandleDeviceLost() override;
        void ClearStallResources() override;

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
        CmdQueue& GetOrCreateCommandQueue(GpuSystem::CmdQueueType type);
        GpuCommandAllocatorInfo& CurrentCommandAllocator(GpuSystem::CmdQueueType type);
        uint64_t ExecuteOnly(GpuCommandList& cmd_list, uint64_t wait_fence_value);
        uint64_t ExecuteOnly(GpuCommandListInternal& cmd_list, uint64_t wait_fence_value);

    private:
        GpuSystem* gpu_system_ = nullptr;

        ComPtr<ID3D12Device> device_;

        CmdQueue cmd_queues_[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)];

        ComPtr<ID3D12Fence> fence_;
        uint64_t fence_val_ = 0;
        Win32UniqueHandle fence_event_;
        Win32UniqueHandle shared_fence_handle_;

        std::list<std::tuple<ComPtr<ID3D12DeviceChild>, uint64_t>> stall_resources_;

        ComPtr<ID3D12CommandSignature> dispatch_indirect_signature_;
    };
} // namespace AIHoloImager
