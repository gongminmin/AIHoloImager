// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <string_view>
#include <type_traits>

#include <directx/d3d12.h>

#include "GpuSystem.hpp"

namespace AIHoloImager
{
    template <typename T>
    class ConstantBuffer final
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);

    public:
        using value_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

    public:
        ConstantBuffer() noexcept = default;

        explicit ConstantBuffer(GpuSystem& gpu_system, uint32_t num_frames = 1, std::wstring_view name = L"")
            : buffer_(gpu_system, num_frames * Align<D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT>(sizeof(value_type)), name),
              aligned_size_(Align<D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT>(sizeof(value_type))), num_frames_(num_frames)
        {
        }

        ConstantBuffer(ConstantBuffer&& other) noexcept
            : buffer_(std::move(other.buffer_)), staging_(std::exchange(other.staging_, value_type{})),
              aligned_size_(std::exchange(other.aligned_size_, 0)), num_frames_(std::exchange(other.num_frames_, 0))
        {
        }

        ConstantBuffer& operator=(ConstantBuffer&& other) noexcept
        {
            if (this != &other)
            {
                buffer_ = std::move(other.buffer_);
                staging_ = std::exchange(other.staging_, value_type{});
                aligned_size_ = std::exchange(other.aligned_size_, 0);
                num_frames_ = std::exchange(other.num_frames_, 0);
            }
            return *this;
        }

        explicit operator bool() const noexcept
        {
            return buffer_ ? true : false;
        }

        const GpuUploadBuffer& Buffer() const noexcept
        {
            return buffer_;
        }

        value_type* MappedData(uint32_t frame_index = 0) noexcept
        {
            return reinterpret_cast<value_type*>(buffer_.MappedData<uint8_t>() + frame_index * aligned_size_);
        }
        const value_type* MappedData(uint32_t frame_index = 0) const noexcept
        {
            return reinterpret_cast<const value_type*>(buffer_.MappedData<uint8_t>() + frame_index * aligned_size_);
        }

        void UploadToGpu(uint32_t frame_index = 0) noexcept
        {
            *this->MappedData(frame_index) = staging_;
        }

        value_type* operator->() noexcept
        {
            return &staging_;
        }
        const value_type* operator->() const noexcept
        {
            return &staging_;
        }

        uint32_t NumFrames() const noexcept
        {
            return num_frames_;
        }

        void* NativeResource() const noexcept
        {
            return buffer_.NativeResource();
        }

        D3D12_GPU_VIRTUAL_ADDRESS GpuVirtualAddress(uint32_t frame_index = 0) const noexcept
        {
            return buffer_.GpuVirtualAddress() + frame_index * aligned_size_;
        }

    private:
        template <uint32_t Alignment>
        constexpr uint32_t Align(uint32_t size) noexcept
        {
            static_assert((Alignment & (Alignment - 1)) == 0);
            return (size + (Alignment - 1)) & ~(Alignment - 1);
        }

    private:
        GpuUploadBuffer buffer_;

        value_type staging_{};

        uint32_t aligned_size_ = Align<D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT>(sizeof(value_type));
        uint32_t num_frames_ = 1;
    };
} // namespace AIHoloImager
