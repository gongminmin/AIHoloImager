// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <string_view>

#include <directx/d3d12.h>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuResource.hpp"

namespace AIHoloImager
{
    class GpuSystem;
    class GpuCommandList;

    struct GpuRange
    {
        uint64_t begin;
        uint64_t end;
    };
    D3D12_RANGE ToD3D12Range(const GpuRange& range);

    class GpuBuffer : public GpuResource
    {
        DISALLOW_COPY_AND_ASSIGN(GpuBuffer)

    public:
        GpuBuffer() noexcept;
        GpuBuffer(GpuSystem& gpu_system, uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name = L"");
        GpuBuffer(GpuSystem& gpu_system, ID3D12Resource* native_resource, GpuResourceState curr_state, std::wstring_view name = L"");
        virtual ~GpuBuffer();

        GpuBuffer(GpuBuffer&& other) noexcept;
        GpuBuffer& operator=(GpuBuffer&& other) noexcept;

        GpuBuffer Share() const;

        ID3D12Resource* NativeBuffer() const noexcept;

        D3D12_GPU_VIRTUAL_ADDRESS GpuVirtualAddress() const noexcept;
        uint32_t Size() const noexcept;

        void* Map(const GpuRange& read_range);
        const void* Map(const GpuRange& read_range) const;
        void* Map();
        const void* Map() const;
        void Unmap(const GpuRange& write_range);
        void Unmap();
        void Unmap() const;

        template <typename T>
        T* Map(const GpuRange& read_range)
        {
            return reinterpret_cast<T*>(this->Map(read_range));
        }
        template <typename T>
        const T* Map(const GpuRange& read_range) const
        {
            return reinterpret_cast<T*>(this->Map(read_range));
        }
        template <typename T>
        T* Map()
        {
            return reinterpret_cast<T*>(this->Map());
        }
        template <typename T>
        const T* Map() const
        {
            return reinterpret_cast<T*>(this->Map());
        }

        virtual void Reset();

        void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const;

    protected:
        D3D12_HEAP_TYPE heap_type_{};
        mutable D3D12_RESOURCE_STATES curr_state_{};
    };
} // namespace AIHoloImager
