// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <mutex>
#include <vector>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuDescriptorHeap.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class GpuDescriptorPage final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDescriptorPage)

    public:
        explicit GpuDescriptorPage(GpuSystem& gpu_system, GpuDescriptorHeapType type, bool shader_visible, uint32_t size);
        ~GpuDescriptorPage() noexcept;

        GpuDescriptorPage(GpuDescriptorPage&& other) noexcept;
        GpuDescriptorPage& operator=(GpuDescriptorPage&& other) noexcept;

        const GpuDescriptorHeap& Heap() const noexcept
        {
            return heap_;
        }

        GpuDescriptorCpuHandle CpuHandleStart() const noexcept
        {
            return cpu_handle_;
        }

        GpuDescriptorGpuHandle GpuHandleStart() const noexcept
        {
            return gpu_handle_;
        }

    private:
        GpuDescriptorHeap heap_;
        GpuDescriptorCpuHandle cpu_handle_;
        GpuDescriptorGpuHandle gpu_handle_;
    };

    class GpuDescriptorBlock final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDescriptorBlock)

    public:
        GpuDescriptorBlock() noexcept;

        GpuDescriptorBlock(GpuDescriptorBlock&& other) noexcept;
        GpuDescriptorBlock& operator=(GpuDescriptorBlock&& other) noexcept;

        void Reset() noexcept;
        void Reset(const GpuDescriptorPage& page, uint32_t offset, uint32_t size) noexcept;

        void* NativeDescriptorHeap() const noexcept
        {
            return native_heap_;
        }
        template <typename Traits>
        typename Traits::DescriptorHeapType NativeDescriptorHeap() const noexcept
        {
            return reinterpret_cast<typename Traits::DescriptorHeapType>(this->NativeDescriptorHeap());
        }

        explicit operator bool() const noexcept
        {
            return (native_heap_ != nullptr);
        }

        uint32_t Offset() const noexcept
        {
            return offset_;
        }

        uint32_t Size() const noexcept
        {
            return size_;
        }

        GpuDescriptorCpuHandle CpuHandle() const noexcept
        {
            return cpu_handle_;
        }

        GpuDescriptorGpuHandle GpuHandle() const noexcept
        {
            return gpu_handle_;
        }

    private:
        void* native_heap_ = nullptr;
        GpuDescriptorHeapType heap_type_{};
        uint32_t offset_ = 0;
        uint32_t size_ = 0;
        GpuDescriptorCpuHandle cpu_handle_{};
        GpuDescriptorGpuHandle gpu_handle_{};
    };

    class GpuDescriptorAllocator final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDescriptorAllocator)

    public:
        GpuDescriptorAllocator(GpuSystem& gpu_system, GpuDescriptorHeapType type, bool shader_visible) noexcept;

        GpuDescriptorAllocator(GpuDescriptorAllocator&& other) noexcept;
        GpuDescriptorAllocator& operator=(GpuDescriptorAllocator&& other) noexcept;

        uint32_t DescriptorSize() const;

        GpuDescriptorBlock Allocate(uint32_t size);
        void Deallocate(GpuDescriptorBlock&& desc_block, uint64_t fence_value);
        void Reallocate(GpuDescriptorBlock& desc_block, uint64_t fence_value, uint32_t size);

        void ClearStallPages(uint64_t fence_value);
        void Clear();

    private:
        void Allocate(std::lock_guard<std::mutex>& proof_of_lock, GpuDescriptorBlock& desc_block, uint32_t size);
        void Deallocate(std::lock_guard<std::mutex>& proof_of_lock, GpuDescriptorBlock& desc_block, uint64_t fence_value);

    private:
        GpuSystem* gpu_system_;
        const GpuDescriptorHeapType type_;
        const bool shader_visible_;

        std::mutex allocation_mutex_;

        struct PageInfo
        {
            GpuDescriptorPage page;

#pragma pack(push, 1)
            struct FreeRange
            {
                uint16_t first_offset;
                uint16_t last_offset;
            };
#pragma pack(pop)
            std::vector<FreeRange> free_list;

#pragma pack(push, 1)
            struct StallRange
            {
                FreeRange free_range;
                uint64_t fence_value;
            };
#pragma pack(pop)
            std::vector<StallRange> stall_list;
        };
        std::vector<PageInfo> pages_;
    };
} // namespace AIHoloImager
