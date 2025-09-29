// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <string_view>
#include <tuple>

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    struct GpuDescriptorCpuHandle
    {
        size_t handle;
    };

    struct GpuDescriptorGpuHandle
    {
        uint64_t handle;
    };

    GpuDescriptorCpuHandle OffsetHandle(const GpuDescriptorCpuHandle& handle, int32_t offset, uint32_t desc_size);
    GpuDescriptorGpuHandle OffsetHandle(const GpuDescriptorGpuHandle& handle, int32_t offset, uint32_t desc_size);
    std::tuple<GpuDescriptorCpuHandle, GpuDescriptorGpuHandle> OffsetHandle(
        const GpuDescriptorCpuHandle& cpu_handle, const GpuDescriptorGpuHandle& gpu_handle, int32_t offset, uint32_t desc_size);

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

        void Name(std::wstring_view name);

        ID3D12DescriptorHeap* NativeDescriptorHeap() const noexcept;

        explicit operator bool() const noexcept;

        GpuDescriptorCpuHandle CpuHandleStart() const noexcept;
        GpuDescriptorGpuHandle GpuHandleStart() const noexcept;

        uint32_t Size() const noexcept;

        void Reset() noexcept;

    private:
        ComPtr<ID3D12DescriptorHeap> heap_;
        D3D12_DESCRIPTOR_HEAP_DESC desc_{};
    };
} // namespace AIHoloImager
