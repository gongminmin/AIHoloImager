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
        void* Map();
        void Unmap(const GpuRange& write_range);
        void Unmap();

        virtual void Reset();

        void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const;

    protected:
        D3D12_HEAP_TYPE heap_type_{};
        mutable D3D12_RESOURCE_STATES curr_state_{};
    };

    class GpuUploadBuffer final : public GpuBuffer
    {
    public:
        GpuUploadBuffer() noexcept;
        GpuUploadBuffer(GpuSystem& gpu_system, uint32_t size, std::wstring_view name = L"");
        GpuUploadBuffer(GpuSystem& gpu_system, const void* data, uint32_t size, std::wstring_view name = L"");
        ~GpuUploadBuffer() noexcept override;

        GpuUploadBuffer(GpuUploadBuffer&& other) noexcept;
        GpuUploadBuffer& operator=(GpuUploadBuffer&& other) noexcept;

        GpuUploadBuffer Share() const;

        void Reset() override;

        void* MappedData() noexcept;

        template <typename T>
        T* MappedData() noexcept
        {
            return reinterpret_cast<T*>(this->MappedData());
        }

    private:
        void* mapped_data_ = nullptr;
    };

    class GpuReadBackBuffer final : public GpuBuffer
    {
    public:
        GpuReadBackBuffer() noexcept;
        GpuReadBackBuffer(GpuSystem& gpu_system, uint32_t size, std::wstring_view name = L"");
        GpuReadBackBuffer(GpuSystem& gpu_system, const void* data, uint32_t size, std::wstring_view name = L"");
        ~GpuReadBackBuffer() noexcept;

        GpuReadBackBuffer(GpuReadBackBuffer&& other) noexcept;
        GpuReadBackBuffer& operator=(GpuReadBackBuffer&& other) noexcept;

        GpuReadBackBuffer Share() const;

        void Reset() override;

        void* MappedData() noexcept;
        const void* MappedData() const noexcept;

        template <typename T>
        T* MappedData() noexcept
        {
            return reinterpret_cast<T*>(this->MappedData());
        }
        template <typename T>
        const T* MappedData() const noexcept
        {
            return reinterpret_cast<const T*>(this->MappedData());
        }

    private:
        void* mapped_data_ = nullptr;
    };
} // namespace AIHoloImager
