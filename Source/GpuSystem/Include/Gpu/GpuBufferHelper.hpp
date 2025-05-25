// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <string>
#include <string_view>
#include <type_traits>

#include <directx/d3d12.h>

#include "Gpu/GpuMemoryAllocator.hpp"
#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    class GeneralConstantBuffer
    {
    public:
        virtual ~GeneralConstantBuffer() noexcept = default;

        virtual D3D12_GPU_VIRTUAL_ADDRESS GpuVirtualAddress() const noexcept = 0;

    protected:
        GeneralConstantBuffer() noexcept = default;
    };

    template <typename T>
    class ConstantBuffer final : public GeneralConstantBuffer
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> && ((sizeof(T) & 0xF) == 0));

    public:
        using value_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

    public:
        ConstantBuffer() noexcept = default;

        explicit ConstantBuffer(GpuSystem& gpu_system, std::wstring_view name = L"")
            : mem_block_(gpu_system.AllocUploadMemBlock(sizeof(value_type), GpuMemoryAllocator::ConstantDataAlignment)),
              name_(std::move(name))
        {
        }

        ConstantBuffer(ConstantBuffer&& other) noexcept
            : mem_block_(std::move(other.mem_block_)), name_(std::move(name_)), staging_(std::exchange(other.staging_, value_type{}))
        {
        }

        ConstantBuffer& operator=(ConstantBuffer&& other) noexcept
        {
            if (this != &other)
            {
                mem_block_ = std::move(other.mem_block_);
                name_ = std::move(name_);
                staging_ = std::exchange(other.staging_, value_type{});
            }
            return *this;
        }

        explicit operator bool() const noexcept
        {
            return mem_block_ ? true : false;
        }

        const GpuMemoryBlock& MemBlock() const noexcept
        {
            return mem_block_;
        }

        value_type* MappedData() noexcept
        {
            return mem_block_.CpuSpan<value_type>().data();
        }
        const value_type* MappedData() const noexcept
        {
            return mem_block_.CpuSpan<value_type>().data();
        }

        void UploadToGpu() noexcept
        {
            *this->MappedData() = staging_;
        }

        value_type* operator->() noexcept
        {
            return &staging_;
        }
        const value_type* operator->() const noexcept
        {
            return &staging_;
        }

        void* NativeResource() const noexcept
        {
            return mem_block_.NativeBuffer();
        }

        D3D12_GPU_VIRTUAL_ADDRESS GpuVirtualAddress() const noexcept override
        {
            return mem_block_.GpuAddress();
        }

    private:
        GpuMemoryBlock mem_block_;
        std::wstring name_;

        value_type staging_{};
    };
} // namespace AIHoloImager
