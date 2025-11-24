// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <functional>
#include <memory>

#include "Base/MiniWindows.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuMemoryAllocator.hpp"
#include "Gpu/GpuMipmapper.hpp"

namespace AIHoloImager
{
    class GpuSystemInternal;
    class GpuSystemInternalFactory;

    class GpuSystem final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuSystem)
        DEFINE_INTERNAL(GpuSystem)

    public:
        enum class Api
        {
            D3D12,
            Vulkan,

            Auto,
        };

        static constexpr uint64_t MaxFenceValue = ~0ull;

        enum class CmdQueueType : uint32_t
        {
            Render = 0,
            Compute,
            VideoEncode,

            Num,
        };

    public:
        GpuSystem(Api api, std::function<bool(Api api, void* device)> confirm_device = nullptr, bool enable_sharing = false,
            bool enable_debug = false);
        ~GpuSystem();

        GpuSystem(GpuSystem&& other) noexcept;
        GpuSystem& operator=(GpuSystem&& other) noexcept;

        Api NativeApi() const noexcept;

        void* NativeDevice() const noexcept;
        template <typename Traits>
        typename Traits::DeviceType NativeDevice() const noexcept
        {
            return reinterpret_cast<typename Traits::DeviceType>(this->NativeDevice());
        }
        void* NativeCommandQueue(CmdQueueType type) const noexcept;
        template <typename Traits>
        typename Traits::CommandQueueType NativeCommandQueue() const noexcept
        {
            return reinterpret_cast<typename Traits::CommandQueueType>(this->NativeCommandQueue());
        }

        LUID DeviceLuid() const noexcept;

        void* SharedFenceHandle() const noexcept;
        template <typename Traits>
        typename Traits::SharedHandleType SharedFenceHandle() const noexcept
        {
            return reinterpret_cast<typename Traits::SharedHandleType>(this->SharedFenceHandle());
        }

        [[nodiscard]] GpuCommandList CreateCommandList(CmdQueueType type);
        uint64_t Execute(GpuCommandList&& cmd_list, uint64_t wait_fence_value = MaxFenceValue);
        uint64_t ExecuteAndReset(GpuCommandList& cmd_list, uint64_t wait_fence_value = MaxFenceValue);

        uint32_t ConstantDataAlignment() const noexcept;
        uint32_t StructuredDataAlignment() const noexcept;
        uint32_t TextureDataAlignment() const noexcept;

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

        GpuMipmapper& Mipmapper() noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    constexpr uint32_t DivUp(uint32_t a, uint32_t b) noexcept
    {
        return (a + b - 1) / b;
    }
} // namespace AIHoloImager
