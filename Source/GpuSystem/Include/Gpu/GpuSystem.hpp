// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <functional>
#include <memory>
#include <span>
#include <string_view>

#include "Base/MiniWindows.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuMipmapper.hpp"
#include "Gpu/Symbol.hpp"

namespace AIHoloImager
{
    class GpuCommandQueue;
    class GpuMemoryBlock;
    class GpuSystemInternal;

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
            Copy,
            VideoEncode,

            Num,
        };

        struct WaitQueueFence
        {
            CmdQueueType type = CmdQueueType::Num;
            uint64_t value = MaxFenceValue;
        };

        struct WaitFences
        {
            uint64_t fence_values[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)] = {};

            AIHI_GPU_SYS_API static WaitFences Forever();
            AIHI_GPU_SYS_API static WaitFences Ignore();
        };

    public:
        AIHI_GPU_SYS_API GpuSystem(Api api, std::function<bool(Api api, void* device)> confirm_device = nullptr,
            bool enable_sharing = false, bool enable_debug = false, bool enable_async_compute = true, bool enable_async_copy = true);
        AIHI_GPU_SYS_API ~GpuSystem();

        AIHI_GPU_SYS_API GpuSystem(GpuSystem&& other) noexcept;
        AIHI_GPU_SYS_API GpuSystem& operator=(GpuSystem&& other) noexcept;

        AIHI_GPU_SYS_API Api NativeApi() const noexcept;

        AIHI_GPU_SYS_API void* NativeDevice() const noexcept;
        template <typename Traits>
        typename Traits::DeviceType NativeDevice() const noexcept
        {
            return reinterpret_cast<typename Traits::DeviceType>(this->NativeDevice());
        }

        AIHI_GPU_SYS_API LUID DeviceLuid() const noexcept;

        AIHI_GPU_SYS_API GpuCommandQueue* CommandQueue(CmdQueueType type) noexcept;
        AIHI_GPU_SYS_API const GpuCommandQueue* CommandQueue(CmdQueueType type) const noexcept;

        AIHI_GPU_SYS_API void* SharedFenceHandle(CmdQueueType type) const noexcept;
        template <typename Traits>
        typename Traits::SharedHandleType SharedFenceHandle() const noexcept
        {
            return reinterpret_cast<typename Traits::SharedHandleType>(this->SharedFenceHandle());
        }

        AIHI_GPU_SYS_API [[nodiscard]] GpuCommandList CreateCommandList(CmdQueueType type, std::string_view name = "");
        AIHI_GPU_SYS_API uint64_t Execute(GpuCommandList&& cmd_list, const WaitFences& wait_fences = WaitFences::Ignore());
        AIHI_GPU_SYS_API uint64_t ExecuteAndReset(GpuCommandList& cmd_list, const WaitFences& wait_fences = WaitFences::Ignore());

        AIHI_GPU_SYS_API uint32_t ConstantDataAlignment() const noexcept;
        AIHI_GPU_SYS_API uint32_t StructuredDataAlignment() const noexcept;
        AIHI_GPU_SYS_API uint32_t TextureDataAlignment() const noexcept;

        AIHI_GPU_SYS_API GpuMemoryBlock AllocUploadMemBlock(uint32_t size_in_bytes, uint32_t alignment);
        AIHI_GPU_SYS_API void DeallocUploadMemBlock(GpuMemoryBlock&& mem_block);
        AIHI_GPU_SYS_API void ReallocUploadMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment);

        AIHI_GPU_SYS_API GpuMemoryBlock AllocReadBackMemBlock(uint32_t size_in_bytes, uint32_t alignment);
        AIHI_GPU_SYS_API void DeallocReadBackMemBlock(GpuMemoryBlock&& mem_block);
        AIHI_GPU_SYS_API void ReallocReadBackMemBlock(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment);

        AIHI_GPU_SYS_API void CpuWait(const WaitFences& wait_fences = WaitFences::Forever());
        AIHI_GPU_SYS_API void GpuWait(CmdQueueType target_queue_type, const WaitFences& wait_fences = WaitFences::Forever());
        AIHI_GPU_SYS_API uint64_t FenceValue(CmdQueueType type) const noexcept;

        AIHI_GPU_SYS_API void HandleDeviceLost();

        AIHI_GPU_SYS_API GpuMipmapper& Mipmapper() noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    constexpr uint32_t DivUp(uint32_t a, uint32_t b) noexcept
    {
        return (a + b - 1) / b;
    }

    AIHI_GPU_SYS_API GpuSystem::WaitFences ToWaitFences(std::span<const GpuSystem::WaitQueueFence> wait_queue_fences);
} // namespace AIHoloImager
