// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <string_view>
#include <tuple>

#include <directx/d3d12.h>

#include "Util/ComPtr.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    D3D12_CPU_DESCRIPTOR_HANDLE OffsetHandle(const D3D12_CPU_DESCRIPTOR_HANDLE& handle, int32_t offset, uint32_t desc_size);
    D3D12_GPU_DESCRIPTOR_HANDLE OffsetHandle(const D3D12_GPU_DESCRIPTOR_HANDLE& handle, int32_t offset, uint32_t desc_size);
    std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE> OffsetHandle(
        const D3D12_CPU_DESCRIPTOR_HANDLE& cpu_handle, const D3D12_GPU_DESCRIPTOR_HANDLE& gpu_handle, int32_t offset, uint32_t desc_size);

    class GpuDescriptorHeap final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDescriptorHeap)

    public:
        GpuDescriptorHeap() noexcept;
        GpuDescriptorHeap(GpuSystem& gpu_system, uint32_t size, D3D12_DESCRIPTOR_HEAP_TYPE type, D3D12_DESCRIPTOR_HEAP_FLAGS flags,
            std::wstring_view name = L"");
        ~GpuDescriptorHeap() noexcept;

        GpuDescriptorHeap(GpuDescriptorHeap&& other) noexcept;
        GpuDescriptorHeap& operator=(GpuDescriptorHeap&& other) noexcept;

        ID3D12DescriptorHeap* NativeDescriptorHeap() const noexcept;

        explicit operator bool() const noexcept;

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandleStart() const noexcept;
        D3D12_GPU_DESCRIPTOR_HANDLE GpuHandleStart() const noexcept;

        uint32_t Size() const noexcept;

        void Reset() noexcept;

    private:
        ComPtr<ID3D12DescriptorHeap> heap_;
        D3D12_DESCRIPTOR_HEAP_DESC desc_{};
    };
} // namespace AIHoloImager
