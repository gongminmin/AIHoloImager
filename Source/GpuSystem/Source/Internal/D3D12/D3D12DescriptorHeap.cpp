// Copyright (c) 2024-2025 Minmin Gong
//

#include "D3D12DescriptorHeap.hpp"

#include "Base/ErrorHandling.hpp"
#include "Base/Uuid.hpp"

#include "D3D12Conversion.hpp"
#include "D3D12System.hpp"
#include "D3D12Util.hpp"

namespace AIHoloImager
{
    D3D12_CPU_DESCRIPTOR_HANDLE OffsetHandle(D3D12_CPU_DESCRIPTOR_HANDLE handle, int32_t offset, uint32_t desc_size)
    {
        return {handle.ptr + offset * desc_size};
    }

    D3D12_GPU_DESCRIPTOR_HANDLE OffsetHandle(D3D12_GPU_DESCRIPTOR_HANDLE handle, int32_t offset, uint32_t desc_size)
    {
        return {handle.ptr + offset * desc_size};
    }

    std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE> OffsetHandle(
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle, int32_t offset, uint32_t desc_size)
    {
        const int32_t offset_in_bytes = offset * desc_size;
        return {{cpu_handle.ptr + offset_in_bytes}, {gpu_handle.ptr + offset_in_bytes}};
    }

    D3D12DescriptorHeap::D3D12DescriptorHeap(
        GpuSystem& gpu_system, uint32_t size, D3D12_DESCRIPTOR_HEAP_TYPE type, bool shader_visible, std::string_view name)
        : type_(type)
    {
        ID3D12Device* d3d12_device = D3D12Imp(gpu_system).Device();

        desc_ = {
            .Type = type,
            .NumDescriptors = size,
            .Flags = shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            .NodeMask = 0,
        };
        TIFHR(d3d12_device->CreateDescriptorHeap(&desc_, UuidOf<ID3D12DescriptorHeap>(), heap_.PutVoid()));
        this->Name(std::move(name));
    }

    D3D12DescriptorHeap::~D3D12DescriptorHeap() noexcept = default;

    D3D12DescriptorHeap::D3D12DescriptorHeap(D3D12DescriptorHeap&& other) noexcept = default;
    D3D12DescriptorHeap& D3D12DescriptorHeap::operator=(D3D12DescriptorHeap&& other) noexcept = default;

    void D3D12DescriptorHeap::Name(std::string_view name)
    {
        SetName(*heap_, std::move(name));
    }

    ID3D12DescriptorHeap* D3D12DescriptorHeap::DescriptorHeap() const noexcept
    {
        return heap_.Get();
    }

    void* D3D12DescriptorHeap::NativeDescriptorHeap() const noexcept
    {
        return this->DescriptorHeap();
    }

    D3D12_DESCRIPTOR_HEAP_TYPE D3D12DescriptorHeap::Type() const noexcept
    {
        return type_;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE D3D12DescriptorHeap::CpuHandleStart() const noexcept
    {
        return heap_->GetCPUDescriptorHandleForHeapStart();
    }

    D3D12_GPU_DESCRIPTOR_HANDLE D3D12DescriptorHeap::GpuHandleStart() const noexcept
    {
        return heap_->GetGPUDescriptorHandleForHeapStart();
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
