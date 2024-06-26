// Copyright (c) 2024 Minmin Gong
//

#include "GpuDescriptorHeap.hpp"

#include "GpuSystem.hpp"
#include "Util/ErrorHandling.hpp"
#include "Util/Uuid.hpp"

namespace AIHoloImager
{
    D3D12_CPU_DESCRIPTOR_HANDLE OffsetHandle(const D3D12_CPU_DESCRIPTOR_HANDLE& handle, int32_t offset, uint32_t desc_size)
    {
        return {handle.ptr + offset * desc_size};
    }

    D3D12_GPU_DESCRIPTOR_HANDLE OffsetHandle(const D3D12_GPU_DESCRIPTOR_HANDLE& handle, int32_t offset, uint32_t desc_size)
    {
        return {handle.ptr + offset * desc_size};
    }

    std::tuple<D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE> OffsetHandle(
        const D3D12_CPU_DESCRIPTOR_HANDLE& cpu_handle, const D3D12_GPU_DESCRIPTOR_HANDLE& gpu_handle, int32_t offset, uint32_t desc_size)
    {
        const int32_t offset_in_bytes = offset * desc_size;
        return {{cpu_handle.ptr + offset_in_bytes}, {gpu_handle.ptr + offset_in_bytes}};
    }


    GpuDescriptorHeap::GpuDescriptorHeap() noexcept = default;

    GpuDescriptorHeap::GpuDescriptorHeap(
        GpuSystem& gpu_system, uint32_t size, D3D12_DESCRIPTOR_HEAP_TYPE type, D3D12_DESCRIPTOR_HEAP_FLAGS flags, std::wstring_view name)
    {
        desc_.Type = type;
        desc_.NumDescriptors = size;
        desc_.Flags = flags;
        desc_.NodeMask = 0;
        TIFHR(gpu_system.NativeDevice()->CreateDescriptorHeap(&desc_, UuidOf<ID3D12DescriptorHeap>(), heap_.PutVoid()));
        if (!name.empty())
        {
            heap_->SetName(std::wstring(std::move(name)).c_str());
        }
    }

    GpuDescriptorHeap::~GpuDescriptorHeap() noexcept = default;
    GpuDescriptorHeap::GpuDescriptorHeap(GpuDescriptorHeap&& other) noexcept = default;
    GpuDescriptorHeap& GpuDescriptorHeap::operator=(GpuDescriptorHeap&& other) noexcept = default;

    GpuDescriptorHeap::operator bool() const noexcept
    {
        return heap_ ? true : false;
    }

    ID3D12DescriptorHeap* GpuDescriptorHeap::NativeDescriptorHeap() const noexcept
    {
        return heap_.Get();
    }

    D3D12_CPU_DESCRIPTOR_HANDLE GpuDescriptorHeap::CpuHandleStart() const noexcept
    {
        return heap_->GetCPUDescriptorHandleForHeapStart();
    }

    D3D12_GPU_DESCRIPTOR_HANDLE GpuDescriptorHeap::GpuHandleStart() const noexcept
    {
        return heap_->GetGPUDescriptorHandleForHeapStart();
    }

    uint32_t GpuDescriptorHeap::Size() const noexcept
    {
        return static_cast<uint32_t>(desc_.NumDescriptors);
    }

    void GpuDescriptorHeap::Reset() noexcept
    {
        heap_ = nullptr;
        desc_ = {};
    }
} // namespace AIHoloImager
