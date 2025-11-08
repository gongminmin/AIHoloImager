// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <memory>
#include <string_view>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuResource.hpp"
#include "Gpu/InternalDefine.hpp"

namespace AIHoloImager
{
    class GpuSystem;
    class GpuCommandList;

    struct GpuRange
    {
        uint64_t begin;
        uint64_t end;
    };

    using GpuVirtualAddressType = uint64_t;

    class GpuBuffer : public GpuResource
    {
        DISALLOW_COPY_AND_ASSIGN(GpuBuffer)
        DEFINE_INTERNAL(GpuResource)

    public:
        GpuBuffer() noexcept;
        GpuBuffer(GpuSystem& gpu_system, uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name = "");
        GpuBuffer(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::string_view name = "");
        ~GpuBuffer() override;

        GpuBuffer(GpuBuffer&& other) noexcept;
        GpuBuffer& operator=(GpuBuffer&& other) noexcept;

        void Name(std::string_view name) override;

        void* NativeResource() const noexcept override;
        template <typename Traits>
        typename Traits::ResourceType NativeResource() const noexcept
        {
            return reinterpret_cast<typename Traits::ResourceType>(this->NativeResource());
        }
        void* NativeBuffer() const noexcept;
        template <typename Traits>
        typename Traits::BufferType NativeBuffer() const noexcept
        {
            return reinterpret_cast<typename Traits::BufferType>(this->NativeBuffer());
        }

        explicit operator bool() const noexcept override;

        void* SharedHandle() const noexcept override;

        GpuResourceType Type() const noexcept override;
        uint32_t AllocationSize() const noexcept override;

        GpuVirtualAddressType GpuVirtualAddress() const noexcept;
        uint32_t Size() const noexcept;

        void* Map(const GpuRange& read_range);
        const void* Map(const GpuRange& read_range) const;
        void* Map();
        const void* Map() const;
        void Unmap(const GpuRange& write_range);
        void Unmap();
        void Unmap() const;

        template <typename T>
        T* Map(const GpuRange& read_range)
        {
            return reinterpret_cast<T*>(this->Map(read_range));
        }
        template <typename T>
        const T* Map(const GpuRange& read_range) const
        {
            return reinterpret_cast<T*>(this->Map(read_range));
        }
        template <typename T>
        T* Map()
        {
            return reinterpret_cast<T*>(this->Map());
        }
        template <typename T>
        const T* Map() const
        {
            return reinterpret_cast<T*>(this->Map());
        }

        void Reset() override;

        void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
