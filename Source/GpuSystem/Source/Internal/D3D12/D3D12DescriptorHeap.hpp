// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <string_view>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    D3D12_CPU_DESCRIPTOR_HANDLE OffsetHandle(D3D12_CPU_DESCRIPTOR_HANDLE handle, int32_t offset, uint32_t desc_size);
    D3D12_GPU_DESCRIPTOR_HANDLE OffsetHandle(D3D12_GPU_DESCRIPTOR_HANDLE handle, int32_t offset, uint32_t desc_size);
    std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE> OffsetHandle(
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle, int32_t offset, uint32_t desc_size);

    class D3D12DescriptorHeap
    {
        DISALLOW_COPY_AND_ASSIGN(D3D12DescriptorHeap)

    public:
        D3D12DescriptorHeap(
            GpuSystem& gpu_system, uint32_t size, D3D12_DESCRIPTOR_HEAP_TYPE type, bool shader_visible, std::string_view name = "");
        ~D3D12DescriptorHeap() noexcept;

        D3D12DescriptorHeap(D3D12DescriptorHeap&& other) noexcept;
        D3D12DescriptorHeap& operator=(D3D12DescriptorHeap&& other) noexcept;

        void Name(std::string_view name);

        ID3D12DescriptorHeap* DescriptorHeap() const noexcept;
        void* NativeDescriptorHeap() const noexcept;

        D3D12_DESCRIPTOR_HEAP_TYPE Type() const noexcept;

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandleStart() const noexcept;
        D3D12_GPU_DESCRIPTOR_HANDLE GpuHandleStart() const noexcept;

        uint32_t Size() const noexcept;

        void Reset() noexcept;

    private:
        ComPtr<ID3D12DescriptorHeap> heap_;
        D3D12_DESCRIPTOR_HEAP_DESC desc_{};
        D3D12_DESCRIPTOR_HEAP_TYPE type_{};
    };
} // namespace AIHoloImager
