// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    class GpuMemoryPage final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuMemoryPage)

    public:
        GpuMemoryPage(GpuSystem& gpu_system, bool is_upload, uint32_t size_in_bytes);
        ~GpuMemoryPage() noexcept;

        GpuMemoryPage(GpuMemoryPage&& other) noexcept;
        GpuMemoryPage& operator=(GpuMemoryPage&& other) noexcept;

        const GpuBuffer& Buffer() const noexcept
        {
            return *buffer_;
        }

        void* CpuAddress() noexcept
        {
            return cpu_addr_;
        }
        const void* CpuAddress() const noexcept
        {
            return cpu_addr_;
        }

        template <typename T>
        T* CpuAddress() noexcept
        {
            return reinterpret_cast<T*>(this->CpuAddress());
        }
        template <typename T>
        const T* CpuAddress() const noexcept
        {
            return reinterpret_cast<const T*>(this->CpuAddress());
        }

    private:
        const bool is_upload_;
        std::unique_ptr<GpuBuffer> buffer_;
        void* cpu_addr_;
    };

    class GpuMemoryBlock final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuMemoryBlock)

    public:
        GpuMemoryBlock() noexcept;
        ~GpuMemoryBlock();

        GpuMemoryBlock(GpuMemoryBlock&& other) noexcept;
        GpuMemoryBlock& operator=(GpuMemoryBlock&& other) noexcept;

        void Reset() noexcept;
        void Reset(GpuMemoryPage& page, uint32_t offset, uint32_t size) noexcept;

        const GpuBuffer* Buffer() const noexcept
        {
            return buffer_;
        }

        explicit operator bool() const noexcept
        {
            return (buffer_ != nullptr);
        }

        uint32_t Offset() const noexcept
        {
            return offset_;
        }

        uint32_t Size() const noexcept
        {
            return size_;
        }

        template <typename T>
        std::span<T> CpuSpan() noexcept
        {
            return std::span<T>(reinterpret_cast<T*>(cpu_addr_), size_ / sizeof(T));
        }
        template <typename T>
        std::span<const T> CpuSpan() const noexcept
        {
            return std::span<const T>(reinterpret_cast<const T*>(cpu_addr_), size_ / sizeof(const T));
        }

    private:
        const GpuBuffer* buffer_ = nullptr;
        uint32_t offset_ = 0;
        uint32_t size_ = 0;
        void* cpu_addr_ = nullptr;
    };

    class GpuMemoryAllocator final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuMemoryAllocator)

    public:
        GpuMemoryAllocator(GpuSystem& gpu_system, bool is_upload) noexcept;
        ~GpuMemoryAllocator();

        GpuMemoryAllocator(GpuMemoryAllocator&& other) noexcept;
        GpuMemoryAllocator& operator=(GpuMemoryAllocator&& other) noexcept;

        GpuMemoryBlock Allocate(uint32_t size_in_bytes, uint32_t alignment);
        void Deallocate(GpuMemoryBlock&& mem_block);
        void Reallocate(GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment);

        void ClearStallPages(GpuSystem::CmdQueueType queue_type, uint64_t completed_fence_value);
        void Clear();

    private:
        void Allocate(std::lock_guard<std::mutex>& proof_of_lock, GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment);
        void Deallocate(std::lock_guard<std::mutex>& proof_of_lock, GpuMemoryBlock& mem_block);

    private:
        GpuSystem* gpu_system_;
        const bool is_upload_;

        std::mutex allocation_mutex_;

        struct PageInfo
        {
            GpuMemoryPage page;

#pragma pack(push, 1)
            struct FreeRange
            {
                uint32_t first_offset;
                uint32_t last_offset;
            };
#pragma pack(pop)
            std::vector<FreeRange> free_list;

#pragma pack(push, 1)
            struct StallRange
            {
                FreeRange free_range;
                uint64_t fence_values[static_cast<uint32_t>(GpuSystem::CmdQueueType::Num)];
            };
#pragma pack(pop)
            std::vector<StallRange> stall_list;
        };
        std::vector<PageInfo> pages_;

        std::vector<GpuMemoryPage> large_pages_;
    };
} // namespace AIHoloImager
