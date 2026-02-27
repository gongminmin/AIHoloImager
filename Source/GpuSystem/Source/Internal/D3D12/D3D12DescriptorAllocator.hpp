// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "Base/Noncopyable.hpp"

#include "D3D12DescriptorHeap.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class D3D12DescriptorPage final
    {
        DISALLOW_COPY_AND_ASSIGN(D3D12DescriptorPage)

    public:
        D3D12DescriptorPage(GpuSystem& gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE type, bool shader_visible, uint32_t size);
        ~D3D12DescriptorPage() noexcept;

        D3D12DescriptorPage(D3D12DescriptorPage&& other) noexcept;
        D3D12DescriptorPage& operator=(D3D12DescriptorPage&& other) noexcept;

        const D3D12DescriptorHeap& Heap() const noexcept
        {
            return *heap_;
        }

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandleStart() const noexcept
        {
            return cpu_handle_;
        }

        D3D12_GPU_DESCRIPTOR_HANDLE GpuHandleStart() const noexcept
        {
            return gpu_handle_;
        }

    private:
        std::unique_ptr<D3D12DescriptorHeap> heap_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_;
        D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_;
    };

    class D3D12DescriptorBlock final
    {
        DISALLOW_COPY_AND_ASSIGN(D3D12DescriptorBlock)

    public:
        D3D12DescriptorBlock() noexcept;
        ~D3D12DescriptorBlock() noexcept;

        D3D12DescriptorBlock(D3D12DescriptorBlock&& other) noexcept;
        D3D12DescriptorBlock& operator=(D3D12DescriptorBlock&& other) noexcept;

        void Reset() noexcept;
        void Reset(const D3D12DescriptorPage& page, uint32_t offset, uint32_t size) noexcept;

        const D3D12DescriptorHeap* Heap() const noexcept
        {
            return heap_;
        }

        explicit operator bool() const noexcept
        {
            return (heap_ != nullptr);
        }

        uint32_t Offset() const noexcept
        {
            return offset_;
        }

        uint32_t Size() const noexcept
        {
            return size_;
        }

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept
        {
            return cpu_handle_;
        }

        D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle() const noexcept
        {
            return gpu_handle_;
        }

    private:
        const D3D12DescriptorHeap* heap_ = nullptr;
        uint32_t offset_ = 0;
        uint32_t size_ = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
        D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_{};
    };

    class D3D12DescriptorAllocator final
    {
        DISALLOW_COPY_AND_ASSIGN(D3D12DescriptorAllocator)

    public:
        D3D12DescriptorAllocator(GpuSystem& gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE type, bool shader_visible) noexcept;
        ~D3D12DescriptorAllocator();

        D3D12DescriptorAllocator(D3D12DescriptorAllocator&& other) noexcept;
        D3D12DescriptorAllocator& operator=(D3D12DescriptorAllocator&& other) noexcept;

        uint32_t DescriptorSize() const;

        D3D12DescriptorBlock Allocate(uint32_t size);
        void Deallocate(D3D12DescriptorBlock&& desc_block);
        void Reallocate(D3D12DescriptorBlock& desc_block, uint32_t size);

        void ClearStallPages(uint64_t completed_fence_value);
        void Clear();

    private:
        void Allocate(std::lock_guard<std::mutex>& proof_of_lock, D3D12DescriptorBlock& desc_block, uint32_t size);
        void Deallocate(std::lock_guard<std::mutex>& proof_of_lock, D3D12DescriptorBlock& desc_block);

    private:
        GpuSystem* gpu_system_;
        const D3D12_DESCRIPTOR_HEAP_TYPE type_;
        const bool shader_visible_;

        std::mutex allocation_mutex_;

        struct PageInfo
        {
            D3D12DescriptorPage page;

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
