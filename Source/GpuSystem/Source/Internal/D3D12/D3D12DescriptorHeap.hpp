// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <string_view>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Gpu/GpuDescriptorHeap.hpp"

#include "../GpuDescriptorHeapInternal.hpp"

namespace AIHoloImager
{
    class D3D12DescriptorHeap : public GpuDescriptorHeapInternal
    {
    public:
        D3D12DescriptorHeap(
            GpuSystem& gpu_system, uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name = L"");
        ~D3D12DescriptorHeap() noexcept;

        D3D12DescriptorHeap(D3D12DescriptorHeap&& other) noexcept;
        explicit D3D12DescriptorHeap(GpuDescriptorHeapInternal&& other) noexcept;

        D3D12DescriptorHeap& operator=(D3D12DescriptorHeap&& other) noexcept;
        GpuDescriptorHeapInternal& operator=(GpuDescriptorHeapInternal&& other) noexcept override;

        void Name(std::wstring_view name) override;

        void* NativeDescriptorHeap() const noexcept override;

        GpuDescriptorHeapType Type() const noexcept override;

        GpuDescriptorCpuHandle CpuHandleStart() const noexcept override;
        GpuDescriptorGpuHandle GpuHandleStart() const noexcept override;

        uint32_t Size() const noexcept override;

        void Reset() noexcept override;

    private:
        ComPtr<ID3D12DescriptorHeap> heap_;
        D3D12_DESCRIPTOR_HEAP_DESC desc_{};
        GpuDescriptorHeapType type_{};
    };
} // namespace AIHoloImager
