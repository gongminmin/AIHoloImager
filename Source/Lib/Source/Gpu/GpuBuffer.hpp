// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <string_view>

#include <directx/d3d12.h>

#include "Util/ComPtr.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuSystem;
    class GpuCommandList;

    class GpuBuffer
    {
        DISALLOW_COPY_AND_ASSIGN(GpuBuffer)

    public:
        GpuBuffer() noexcept;
        GpuBuffer(GpuSystem& gpu_system, uint32_t size, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_FLAGS flags,
            D3D12_RESOURCE_STATES init_state, std::wstring_view name = L"");
        GpuBuffer(ID3D12Resource* native_resource, D3D12_RESOURCE_STATES curr_state, std::wstring_view name = L"");
        virtual ~GpuBuffer() noexcept;

        GpuBuffer(GpuBuffer&& other) noexcept;
        GpuBuffer& operator=(GpuBuffer&& other) noexcept;

        GpuBuffer Share() const;

        explicit operator bool() const noexcept;

        ID3D12Resource* NativeBuffer() const noexcept;

        D3D12_GPU_VIRTUAL_ADDRESS GpuVirtualAddress() const noexcept;
        uint32_t Size() const noexcept;

        void* Map(const D3D12_RANGE& read_range);
        void* Map();
        void Unmap(const D3D12_RANGE& write_range);
        void Unmap();

        virtual void Reset() noexcept;

        D3D12_RESOURCE_STATES State() const noexcept;
        void Transition(GpuCommandList& cmd_list, D3D12_RESOURCE_STATES target_state);

    protected:
        ComPtr<ID3D12Resource> resource_;
        D3D12_RESOURCE_DESC desc_{};
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

        void Reset() noexcept override;

        void* MappedData() noexcept;

        template <typename T>
        T* MappedData() noexcept
        {
            return reinterpret_cast<T*>(this->MappedData());
        }

    private:
        void* mapped_data_ = nullptr;
    };
} // namespace AIHoloImager
