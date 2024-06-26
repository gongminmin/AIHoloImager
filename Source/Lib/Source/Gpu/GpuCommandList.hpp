// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <span>

#include "GpuSystem.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuCommandList
    {
        DISALLOW_COPY_AND_ASSIGN(GpuCommandList)

    public:
        GpuCommandList() noexcept;
        GpuCommandList(GpuSystem& gpu_system, ID3D12CommandAllocator* cmd_allocator, GpuSystem::CmdQueueType type);
        ~GpuCommandList() noexcept;

        GpuCommandList(GpuCommandList&& other) noexcept;
        GpuCommandList& operator=(GpuCommandList&& other) noexcept;

        GpuSystem::CmdQueueType Type() const noexcept;

        explicit operator bool() const noexcept;

        ID3D12CommandList* NativeCommandListBase() const noexcept
        {
            return cmd_list_.Get();
        }
        template <typename T>
        T* NativeCommandList() const noexcept
        {
            return static_cast<T*>(NativeCommandListBase());
        }

        void Transition(std::span<const D3D12_RESOURCE_BARRIER> barriers) const noexcept;

        void Close();
        void Reset(ID3D12CommandAllocator* cmd_allocator);

    private:
        GpuSystem::CmdQueueType type_ = GpuSystem::CmdQueueType::Num;
        ComPtr<ID3D12CommandList> cmd_list_;
    };
} // namespace AIHoloImager
