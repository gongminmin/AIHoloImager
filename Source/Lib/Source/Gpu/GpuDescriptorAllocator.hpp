// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <mutex>
#include <vector>

#include <directx/d3d12.h>

#include "GpuDescriptorHeap.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class GpuDescriptorPage final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDescriptorPage)

    public:
        explicit GpuDescriptorPage(
            GpuSystem& gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE type, D3D12_DESCRIPTOR_HEAP_FLAGS flags, uint32_t size);
        ~GpuDescriptorPage() noexcept;

        GpuDescriptorPage(GpuDescriptorPage&& other) noexcept;
        GpuDescriptorPage& operator=(GpuDescriptorPage&& other) noexcept;

        const GpuDescriptorHeap& Heap() const noexcept
        {
            return heap_;
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
        GpuDescriptorHeap heap_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_;
        D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_;
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

        ID3D12DescriptorHeap* NativeDescriptorHeap() const noexcept
        {
            return native_heap_;
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

        D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle() const noexcept
        {
            return cpu_handle_;
        }

        D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle() const noexcept
        {
            return gpu_handle_;
        }

    private:
        ID3D12DescriptorHeap* native_heap_ = nullptr;
        uint32_t offset_ = 0;
        uint32_t size_ = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
        D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_{};
    };

    class GpuDescriptorAllocator final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDescriptorAllocator)

    public:
        GpuDescriptorAllocator(GpuSystem& gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE type, D3D12_DESCRIPTOR_HEAP_FLAGS flags) noexcept;

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
        const D3D12_DESCRIPTOR_HEAP_TYPE type_;
        const D3D12_DESCRIPTOR_HEAP_FLAGS flags_;

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
