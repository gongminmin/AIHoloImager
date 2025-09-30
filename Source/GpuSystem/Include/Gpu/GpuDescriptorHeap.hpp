// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <memory>
#include <string_view>
#include <tuple>

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

    enum class GpuDescriptorHeapType
    {
        CbvSrvUav,
        Rtv,
        Dsv,
        Sampler,

        Num,
    };
    constexpr uint32_t NumGpuDescriptorHeapTypes = static_cast<uint32_t>(GpuDescriptorHeapType::Num);

    GpuDescriptorCpuHandle OffsetHandle(const GpuDescriptorCpuHandle& handle, int32_t offset, uint32_t desc_size);
    GpuDescriptorGpuHandle OffsetHandle(const GpuDescriptorGpuHandle& handle, int32_t offset, uint32_t desc_size);
    std::tuple<GpuDescriptorCpuHandle, GpuDescriptorGpuHandle> OffsetHandle(
        const GpuDescriptorCpuHandle& cpu_handle, const GpuDescriptorGpuHandle& gpu_handle, int32_t offset, uint32_t desc_size);

    class GpuDescriptorHeap final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDescriptorHeap)

    public:
        GpuDescriptorHeap() noexcept;
        GpuDescriptorHeap(
            GpuSystem& gpu_system, uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name = L"");
        ~GpuDescriptorHeap() noexcept;

        GpuDescriptorHeap(GpuDescriptorHeap&& other) noexcept;
        GpuDescriptorHeap& operator=(GpuDescriptorHeap&& other) noexcept;

        void Name(std::wstring_view name);

        void* NativeDescriptorHeap() const noexcept;
        template <typename Traits>
        typename Traits::DescriptorHeapType NativeDescriptorHeap() const noexcept
        {
            return static_cast<typename Traits::DescriptorHeapType>(this->NativeDescriptorHeap());
        }

        explicit operator bool() const noexcept;

        GpuDescriptorCpuHandle CpuHandleStart() const noexcept;
        GpuDescriptorGpuHandle GpuHandleStart() const noexcept;

        uint32_t Size() const noexcept;

        void Reset() noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
