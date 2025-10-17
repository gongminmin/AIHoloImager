// Copyright (c) 2025 Minmin Gong
//

#include "D3D12DescriptorHeap.hpp"

#include "Base/ErrorHandling.hpp"
#include "Base/Uuid.hpp"
#include "Gpu/D3D12/D3D12Traits.hpp"

#include "D3D12/D3D12Conversion.hpp"

namespace AIHoloImager
{
    D3D12DescriptorHeap::D3D12DescriptorHeap(
        GpuSystem& gpu_system, uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name)
        : type_(type)
    {
        desc_.Type = ToD3D12DescriptorHeapType(type);
        desc_.NumDescriptors = size;
        desc_.Flags = shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        desc_.NodeMask = 0;
        TIFHR(gpu_system.NativeDevice<D3D12Traits>()->CreateDescriptorHeap(&desc_, UuidOf<ID3D12DescriptorHeap>(), heap_.PutVoid()));
        this->Name(std::move(name));
    }

    D3D12DescriptorHeap::~D3D12DescriptorHeap() noexcept = default;

    D3D12DescriptorHeap::D3D12DescriptorHeap(D3D12DescriptorHeap&& other) noexcept = default;
    D3D12DescriptorHeap::D3D12DescriptorHeap(GpuDescriptorHeapInternal&& other) noexcept
        : D3D12DescriptorHeap(std::forward<D3D12DescriptorHeap>(static_cast<D3D12DescriptorHeap&&>(other)))
    {
    }

    D3D12DescriptorHeap& D3D12DescriptorHeap::operator=(D3D12DescriptorHeap&& other) noexcept = default;
    GpuDescriptorHeapInternal& D3D12DescriptorHeap::operator=(GpuDescriptorHeapInternal&& other) noexcept
    {
        return this->operator=(std::move(static_cast<D3D12DescriptorHeap&&>(other)));
    }

    void D3D12DescriptorHeap::Name(std::wstring_view name)
    {
        heap_->SetName(name.empty() ? L"" : std::wstring(std::move(name)).c_str());
    }

    void* D3D12DescriptorHeap::NativeDescriptorHeap() const noexcept
    {
        return heap_.Get();
    }

    GpuDescriptorHeapType D3D12DescriptorHeap::Type() const noexcept
    {
        return type_;
    }

    GpuDescriptorCpuHandle D3D12DescriptorHeap::CpuHandleStart() const noexcept
    {
        return FromD3D12CpuDescriptorHandle(heap_->GetCPUDescriptorHandleForHeapStart());
    }

    GpuDescriptorGpuHandle D3D12DescriptorHeap::GpuHandleStart() const noexcept
    {
        return FromD3D12GpuDescriptorHandle(heap_->GetGPUDescriptorHandleForHeapStart());
    }

    uint32_t D3D12DescriptorHeap::Size() const noexcept
    {
        return static_cast<uint32_t>(desc_.NumDescriptors);
    }

    void D3D12DescriptorHeap::Reset() noexcept
    {
        heap_ = nullptr;
        desc_ = {};
    }
} // namespace AIHoloImager
