// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <string>
#include <string_view>
#include <type_traits>

#include "Gpu/GpuMemoryAllocator.hpp"
#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    class GpuConstantBuffer
    {
    public:
        virtual ~GpuConstantBuffer();

        explicit operator bool() const noexcept;

        const GpuMemoryBlock& MemBlock() const noexcept;

        void* NativeResource() const noexcept;
        template <typename Traits>
        typename Traits::BufferType NativeResource() const noexcept
        {
            return reinterpret_cast<typename Traits::BufferType>(this->NativeResource());
        }

    protected:
        GpuConstantBuffer() noexcept;
        GpuConstantBuffer(GpuSystem& gpu_system, uint32_t size, std::string_view name = "");

        GpuConstantBuffer(GpuConstantBuffer&& other) noexcept;
        GpuConstantBuffer& operator=(GpuConstantBuffer&& other) noexcept;

    protected:
        GpuSystem* gpu_system_ = nullptr;
        GpuMemoryBlock mem_block_;
        std::string name_;
    };

    template <typename T>
    class GpuConstantBufferOfType final : public GpuConstantBuffer
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> && ((sizeof(T) & 0xF) == 0));

    public:
        using value_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

    public:
        GpuConstantBufferOfType() noexcept = default;
        ~GpuConstantBufferOfType() override = default;

        explicit GpuConstantBufferOfType(GpuSystem& gpu_system, std::string_view name = "")
            : GpuConstantBuffer(gpu_system, sizeof(value_type), std::move(name))
        {
        }

        GpuConstantBufferOfType(GpuConstantBufferOfType&& other) noexcept
            : GpuConstantBuffer(other), staging_(std::exchange(other.staging_, value_type{}))
        {
        }

        GpuConstantBufferOfType& operator=(GpuConstantBufferOfType&& other) noexcept
        {
            if (this != &other)
            {
                staging_ = std::exchange(other.staging_, value_type{});
                GpuConstantBuffer::operator=(std::move(other));
            }
            return *this;
        }

        value_type* MappedData() noexcept
        {
            return mem_block_.CpuSpan<value_type>().data();
        }
        const value_type* MappedData() const noexcept
        {
            return mem_block_.CpuSpan<value_type>().data();
        }

        void UploadStaging() noexcept
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

    private:
        value_type staging_{};
    };
} // namespace AIHoloImager
