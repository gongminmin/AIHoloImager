// Copyright (c) 2025 Minmin Gong
//

#include "D3D12DescriptorHeap.hpp"

#include "Base/ErrorHandling.hpp"
#include "Base/Uuid.hpp"

#include "D3D12Conversion.hpp"
#include "D3D12System.hpp"

namespace AIHoloImager
{
    D3D12_IMP_IMP(DescriptorHeap)

    D3D12DescriptorHeap::D3D12DescriptorHeap(
        GpuSystem& gpu_system, uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name)
        : type_(type)
    {
        ID3D12Device* d3d12_device = D3D12Imp(gpu_system).Device();

        desc_.Type = ToD3D12DescriptorHeapType(type);
        desc_.NumDescriptors = size;
        desc_.Flags = shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        desc_.NodeMask = 0;
        TIFHR(d3d12_device->CreateDescriptorHeap(&desc_, UuidOf<ID3D12DescriptorHeap>(), heap_.PutVoid()));
        this->Name(std::move(name));
    }

    D3D12DescriptorHeap::~D3D12DescriptorHeap() noexcept = default;

    D3D12DescriptorHeap::D3D12DescriptorHeap(D3D12DescriptorHeap&& other) noexcept = default;
    D3D12DescriptorHeap::D3D12DescriptorHeap(GpuDescriptorHeapInternal&& other) noexcept
        : D3D12DescriptorHeap(static_cast<D3D12DescriptorHeap&&>(other))
    {
    }

    D3D12DescriptorHeap& D3D12DescriptorHeap::operator=(D3D12DescriptorHeap&& other) noexcept = default;
    GpuDescriptorHeapInternal& D3D12DescriptorHeap::operator=(GpuDescriptorHeapInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12DescriptorHeap&&>(other));
    }

    void D3D12DescriptorHeap::Name(std::wstring_view name)
    {
        heap_->SetName(name.empty() ? L"" : std::wstring(std::move(name)).c_str());
    }

    ID3D12DescriptorHeap* D3D12DescriptorHeap::DescriptorHeap() const noexcept
    {
        return heap_.Get();
    }

    void* D3D12DescriptorHeap::NativeDescriptorHeap() const noexcept
    {
        return this->DescriptorHeap();
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
