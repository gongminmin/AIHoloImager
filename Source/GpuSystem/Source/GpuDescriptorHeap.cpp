// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuDescriptorHeap.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
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


    class GpuDescriptorHeap::Impl
    {
    public:
        Impl() noexcept = default;
        Impl(GpuSystem& gpu_system, uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name)
        {
            desc_.Type = ToD3D12DescriptorHeapType(type);
            desc_.NumDescriptors = size;
            desc_.Flags = shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
            desc_.NodeMask = 0;
            TIFHR(gpu_system.NativeDevice()->CreateDescriptorHeap(&desc_, UuidOf<ID3D12DescriptorHeap>(), heap_.PutVoid()));
            this->Name(std::move(name));
        }

        ~Impl() noexcept = default;

        Impl(Impl&& other) noexcept = default;
        Impl& operator=(Impl&& other) noexcept = default;

        void Name(std::wstring_view name)
        {
            heap_->SetName(name.empty() ? L"" : std::wstring(std::move(name)).c_str());
        }

        explicit operator bool() const noexcept
        {
            return heap_ ? true : false;
        }

        void* NativeDescriptorHeap() const noexcept
        {
            return heap_.Get();
        }

        GpuDescriptorCpuHandle CpuHandleStart() const noexcept
        {
            return FromD3D12CpuDescriptorHandle(heap_->GetCPUDescriptorHandleForHeapStart());
        }

        GpuDescriptorGpuHandle GpuHandleStart() const noexcept
        {
            return FromD3D12GpuDescriptorHandle(heap_->GetGPUDescriptorHandleForHeapStart());
        }

        uint32_t Size() const noexcept
        {
            return static_cast<uint32_t>(desc_.NumDescriptors);
        }

        void Reset() noexcept
        {
            heap_ = nullptr;
            desc_ = {};
        }

    private:
        ComPtr<ID3D12DescriptorHeap> heap_;
        D3D12_DESCRIPTOR_HEAP_DESC desc_{};
    };


    GpuDescriptorHeap::GpuDescriptorHeap() noexcept : impl_(std::make_unique<Impl>())
    {
    }

    GpuDescriptorHeap::GpuDescriptorHeap(
        GpuSystem& gpu_system, uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name)
        : impl_(std::make_unique<Impl>(gpu_system, size, type, shader_visible, std::move(name)))
    {
    }

    GpuDescriptorHeap::~GpuDescriptorHeap() noexcept = default;
    GpuDescriptorHeap::GpuDescriptorHeap(GpuDescriptorHeap&& other) noexcept = default;
    GpuDescriptorHeap& GpuDescriptorHeap::operator=(GpuDescriptorHeap&& other) noexcept = default;

    void GpuDescriptorHeap::Name(std::wstring_view name)
    {
        assert(impl_);
        impl_->Name(std::move(name));
    }

    GpuDescriptorHeap::operator bool() const noexcept
    {
        return impl_ && impl_->operator bool();
    }

    void* GpuDescriptorHeap::NativeDescriptorHeap() const noexcept
    {
        assert(impl_);
        return impl_->NativeDescriptorHeap();
    }

    GpuDescriptorCpuHandle GpuDescriptorHeap::CpuHandleStart() const noexcept
    {
        assert(impl_);
        return impl_->CpuHandleStart();
    }

    GpuDescriptorGpuHandle GpuDescriptorHeap::GpuHandleStart() const noexcept
    {
        assert(impl_);
        return impl_->GpuHandleStart();
    }

    uint32_t GpuDescriptorHeap::Size() const noexcept
    {
        assert(impl_);
        return impl_->Size();
    }

    void GpuDescriptorHeap::Reset() noexcept
    {
        assert(impl_);
        impl_->Reset();
    }
} // namespace AIHoloImager
