// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuDescriptorHeap.hpp"

#include "Base/ErrorHandling.hpp"
#include "Base/Uuid.hpp"
#include "Gpu/GpuSystem.hpp"

#include "D3D12/D3D12Conversion.hpp"

namespace AIHoloImager
{
    GpuDescriptorCpuHandle OffsetHandle(const GpuDescriptorCpuHandle& handle, int32_t offset, uint32_t desc_size)
    {
        return {handle.handle + offset * desc_size};
    }

    GpuDescriptorGpuHandle OffsetHandle(const GpuDescriptorGpuHandle& handle, int32_t offset, uint32_t desc_size)
    {
        return {handle.handle + offset * desc_size};
    }

    std::tuple<GpuDescriptorCpuHandle, GpuDescriptorGpuHandle> OffsetHandle(
        const GpuDescriptorCpuHandle& cpu_handle, const GpuDescriptorGpuHandle& gpu_handle, int32_t offset, uint32_t desc_size)
    {
        const int32_t offset_in_bytes = offset * desc_size;
        return {{cpu_handle.handle + offset_in_bytes}, {gpu_handle.handle + offset_in_bytes}};
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
        this->Name(std::move(name));
    }

    GpuDescriptorHeap::~GpuDescriptorHeap() noexcept = default;
    GpuDescriptorHeap::GpuDescriptorHeap(GpuDescriptorHeap&& other) noexcept = default;
    GpuDescriptorHeap& GpuDescriptorHeap::operator=(GpuDescriptorHeap&& other) noexcept = default;

    void GpuDescriptorHeap::Name(std::wstring_view name)
    {
        heap_->SetName(name.empty() ? L"" : std::wstring(std::move(name)).c_str());
    }

    GpuDescriptorHeap::operator bool() const noexcept
    {
        return heap_ ? true : false;
    }

    ID3D12DescriptorHeap* GpuDescriptorHeap::NativeDescriptorHeap() const noexcept
    {
        return heap_.Get();
    }

    GpuDescriptorCpuHandle GpuDescriptorHeap::CpuHandleStart() const noexcept
    {
        return FromD3D12CpuDescriptorHandle(heap_->GetCPUDescriptorHandleForHeapStart());
    }

    GpuDescriptorGpuHandle GpuDescriptorHeap::GpuHandleStart() const noexcept
    {
        return FromD3D12GpuDescriptorHandle(heap_->GetGPUDescriptorHandleForHeapStart());
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
