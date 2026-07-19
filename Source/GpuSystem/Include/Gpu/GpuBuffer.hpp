// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <memory>
#include <string_view>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuResource.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/InternalDefine.hpp"
#include "Gpu/Symbol.hpp"

namespace AIHoloImager
{
    class GpuCommandList;

    struct GpuRange
    {
        uint64_t begin;
        uint64_t end;
    };

    class GpuBuffer : public GpuResource
    {
        DISALLOW_COPY_AND_ASSIGN(GpuBuffer)
        DEFINE_INTERNAL(GpuResource)

    public:
        AIHI_GPU_SYS_API GpuBuffer() noexcept;
        AIHI_GPU_SYS_API GpuBuffer(GpuSystem& gpu_system, uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::string_view name = "");
        AIHI_GPU_SYS_API ~GpuBuffer() override;

        AIHI_GPU_SYS_API GpuBuffer(GpuBuffer&& other) noexcept;
        AIHI_GPU_SYS_API GpuBuffer& operator=(GpuBuffer&& other) noexcept;

        AIHI_GPU_SYS_API void Name(std::string_view name) override;

        AIHI_GPU_SYS_API void* NativeResource() const noexcept override;
        template <typename Traits>
        typename Traits::ResourceType NativeResource() const noexcept
        {
            return reinterpret_cast<typename Traits::ResourceType>(this->NativeResource());
        }
        AIHI_GPU_SYS_API void* NativeBuffer() const noexcept;
        template <typename Traits>
        typename Traits::BufferType NativeBuffer() const noexcept
        {
            return reinterpret_cast<typename Traits::BufferType>(this->NativeBuffer());
        }

        AIHI_GPU_SYS_API explicit operator bool() const noexcept override;

        AIHI_GPU_SYS_API void* SharedHandle() const noexcept override;

        AIHI_GPU_SYS_API GpuHeap Heap() const noexcept override;
        AIHI_GPU_SYS_API GpuResourceType Type() const noexcept override;
        AIHI_GPU_SYS_API GpuResourceFlag Flags() const noexcept override;
        AIHI_GPU_SYS_API uint32_t AllocationSize() const noexcept override;

        AIHI_GPU_SYS_API uint32_t Size() const noexcept;

        AIHI_GPU_SYS_API void* Map(const GpuRange& read_range);
        AIHI_GPU_SYS_API const void* Map(const GpuRange& read_range) const;
        AIHI_GPU_SYS_API void* Map();
        AIHI_GPU_SYS_API const void* Map() const;
        AIHI_GPU_SYS_API void Unmap(const GpuRange& write_range);
        AIHI_GPU_SYS_API void Unmap();
        AIHI_GPU_SYS_API void Unmap() const;

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

        AIHI_GPU_SYS_API void Reset() override;

        AIHI_GPU_SYS_API void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        AIHI_GPU_SYS_API void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const override;

        AIHI_GPU_SYS_API const std::shared_ptr<GpuSystem::WaitFences>& StalledWaitFences() const noexcept override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
