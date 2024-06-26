// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <mutex>
#include <vector>

#include "GpuBuffer.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuSystem;

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
            return buffer_;
        }

        void* CpuAddress() const noexcept
        {
            return cpu_addr_;
        }

        template <typename T>
        T* CpuAddress() const noexcept
        {
            return reinterpret_cast<T*>(this->CpuAddress());
        }

        D3D12_GPU_VIRTUAL_ADDRESS GpuAddress() const noexcept
        {
            return gpu_addr_;
        }

    private:
        const bool is_upload_;
        GpuBuffer buffer_;
        void* cpu_addr_;
        D3D12_GPU_VIRTUAL_ADDRESS gpu_addr_;
    };

    class GpuMemoryBlock final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuMemoryBlock)

    public:
        GpuMemoryBlock() noexcept;

        GpuMemoryBlock(GpuMemoryBlock&& other) noexcept;
        GpuMemoryBlock& operator=(GpuMemoryBlock&& other) noexcept;

        void Reset() noexcept;
        void Reset(const GpuMemoryPage& page, uint32_t offset, uint32_t size) noexcept;

        ID3D12Resource* NativeBuffer() const noexcept
        {
            return native_buffer_;
        }

        explicit operator bool() const noexcept
        {
            return (native_buffer_ != nullptr);
        }

        uint32_t Offset() const noexcept
        {
            return offset_;
        }

        uint32_t Size() const noexcept
        {
            return size_;
        }

        void* CpuAddress() const noexcept
        {
            return cpu_addr_;
        }

        template <typename T>
        T* CpuAddress() const noexcept
        {
            return reinterpret_cast<T*>(this->CpuAddress());
        }

        D3D12_GPU_VIRTUAL_ADDRESS GpuAddress() const noexcept
        {
            return gpu_addr_;
        }

    private:
        ID3D12Resource* native_buffer_ = nullptr;
        uint32_t offset_ = 0;
        uint32_t size_ = 0;
        void* cpu_addr_ = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS gpu_addr_ = 0;
    };

    class GpuMemoryAllocator final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuMemoryAllocator)

    public:
        static constexpr uint32_t ConstantDataAligment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
        static constexpr uint32_t StructuredDataAligment = D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT;
        static constexpr uint32_t TextureDataAligment = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;

    public:
        GpuMemoryAllocator(GpuSystem& gpu_system, bool is_upload) noexcept;

        GpuMemoryAllocator(GpuMemoryAllocator&& other) noexcept;
        GpuMemoryAllocator& operator=(GpuMemoryAllocator&& other) noexcept;

        GpuMemoryBlock Allocate(uint32_t size_in_bytes, uint32_t alignment);
        void Deallocate(GpuMemoryBlock&& mem_block, uint64_t fence_value);
        void Reallocate(GpuMemoryBlock& mem_block, uint64_t fence_value, uint32_t size_in_bytes, uint32_t alignment);

        void ClearStallPages(uint64_t fence_value);
        void Clear();

    private:
        void Allocate(std::lock_guard<std::mutex>& proof_of_lock, GpuMemoryBlock& mem_block, uint32_t size_in_bytes, uint32_t alignment);
        void Deallocate(std::lock_guard<std::mutex>& proof_of_lock, GpuMemoryBlock& mem_block, uint64_t fence_value);

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
                uint64_t fence_value;
            };
#pragma pack(pop)
            std::vector<StallRange> stall_list;
        };
        std::vector<PageInfo> pages_;

        std::vector<GpuMemoryPage> large_pages_;
    };
} // namespace AIHoloImager
