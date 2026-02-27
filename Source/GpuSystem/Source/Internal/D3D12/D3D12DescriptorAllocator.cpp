// Copyright (c) 2024-2026 Minmin Gong
//

#include "D3D12DescriptorAllocator.hpp"

#include <cassert>

#include "Base/ErrorHandling.hpp"
#include "Gpu/GpuSystem.hpp"

#include "D3D12System.hpp"
#include "Internal/GpuSystemInternal.hpp"

using namespace AIHoloImager;

namespace
{
    uint32_t descriptor_size[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES]{};

    constexpr uint16_t DescriptorPageSize[] = {32 * 1024, 1 * 1024, 8 * 1024, 4 * 1024};

    uint32_t& DescriptorSize(D3D12_DESCRIPTOR_HEAP_TYPE type)
    {
        return descriptor_size[type];
    }

    void UpdateDescriptorSize(GpuSystem& gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE type)
    {
        auto& size = DescriptorSize(type);
        if (size == 0)
        {
            size = D3D12Imp(gpu_system).DescriptorSize(type);
        }
    }
} // namespace

namespace AIHoloImager
{
    D3D12DescriptorPage::D3D12DescriptorPage(GpuSystem& gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE type, bool shader_visible, uint32_t size)
    {
        heap_ = std::make_unique<D3D12DescriptorHeap>(gpu_system, size, type, shader_visible, "GpuDescriptorPage");
        cpu_handle_ = heap_->CpuHandleStart();
        if (shader_visible)
        {
            gpu_handle_ = heap_->GpuHandleStart();
        }
        else
        {
            gpu_handle_ = {};
        }
    }

    D3D12DescriptorPage::~D3D12DescriptorPage() noexcept = default;
    D3D12DescriptorPage::D3D12DescriptorPage(D3D12DescriptorPage&& other) noexcept = default;
    D3D12DescriptorPage& D3D12DescriptorPage::operator=(D3D12DescriptorPage&& other) noexcept = default;


    D3D12DescriptorBlock::D3D12DescriptorBlock() noexcept = default;
    D3D12DescriptorBlock::~D3D12DescriptorBlock() noexcept = default;

    D3D12DescriptorBlock::D3D12DescriptorBlock(D3D12DescriptorBlock&& other) noexcept = default;
    D3D12DescriptorBlock& D3D12DescriptorBlock::operator=(D3D12DescriptorBlock&& other) noexcept = default;

    void D3D12DescriptorBlock::Reset() noexcept
    {
        heap_ = nullptr;
        offset_ = 0;
        size_ = 0;
        cpu_handle_ = {};
        gpu_handle_ = {};
    }

    void D3D12DescriptorBlock::Reset(const D3D12DescriptorPage& page, uint32_t offset, uint32_t size) noexcept
    {
        heap_ = &page.Heap();
        offset_ = offset;
        size_ = size;

        const uint32_t desc_size = DescriptorSize(heap_->Type());
        std::tie(cpu_handle_, gpu_handle_) = OffsetHandle(page.CpuHandleStart(), page.GpuHandleStart(), offset, desc_size);
    }


    D3D12DescriptorAllocator::D3D12DescriptorAllocator(GpuSystem& gpu_system, D3D12_DESCRIPTOR_HEAP_TYPE type, bool shader_visible) noexcept
        : gpu_system_(&gpu_system), type_(type), shader_visible_(shader_visible)
    {
    }

    D3D12DescriptorAllocator::~D3D12DescriptorAllocator() = default;

    D3D12DescriptorAllocator::D3D12DescriptorAllocator(D3D12DescriptorAllocator&& other) noexcept
        : gpu_system_(std::exchange(other.gpu_system_, {})), type_(other.type_), shader_visible_(other.shader_visible_),
          pages_(std::move(other.pages_))
    {
    }

    D3D12DescriptorAllocator& D3D12DescriptorAllocator::operator=(D3D12DescriptorAllocator&& other) noexcept
    {
        if (this != &other)
        {
            assert(type_ == other.type_);
            assert(shader_visible_ == other.shader_visible_);

            gpu_system_ = std::exchange(other.gpu_system_, {});
            pages_ = std::move(other.pages_);
        }
        return *this;
    }

    uint32_t D3D12DescriptorAllocator::DescriptorSize() const
    {
        UpdateDescriptorSize(*gpu_system_, type_);
        return ::DescriptorSize(type_);
    }

    D3D12DescriptorBlock D3D12DescriptorAllocator::Allocate(uint32_t size)
    {
        std::lock_guard<std::mutex> lock(allocation_mutex_);

        D3D12DescriptorBlock desc_block;
        this->Allocate(lock, desc_block, size);
        return desc_block;
    }

    void D3D12DescriptorAllocator::Allocate(
        [[maybe_unused]] std::lock_guard<std::mutex>& proof_of_lock, D3D12DescriptorBlock& desc_block, uint32_t size)
    {
        UpdateDescriptorSize(*gpu_system_, type_);

        for (auto& page_info : pages_)
        {
            auto const iter = std::lower_bound(page_info.free_list.begin(), page_info.free_list.end(), size,
                [](PageInfo::FreeRange const& free_range, uint32_t s) { return free_range.first_offset + s > free_range.last_offset; });
            if (iter != page_info.free_list.end())
            {
                desc_block.Reset(page_info.page, iter->first_offset, size);
                iter->first_offset += static_cast<uint16_t>(size);
                if (iter->first_offset == iter->last_offset)
                {
                    page_info.free_list.erase(iter);
                }

                return;
            }
        }

        const uint16_t default_page_size = DescriptorPageSize[type_];
        assert(size <= default_page_size);

        D3D12DescriptorPage new_page(*gpu_system_, type_, shader_visible_, default_page_size);
        desc_block.Reset(new_page, 0, size);
        pages_.emplace_back(PageInfo{std::move(new_page), {{static_cast<uint16_t>(size), default_page_size}}, {}});
    }

    void D3D12DescriptorAllocator::Deallocate(D3D12DescriptorBlock&& desc_block)
    {
        if (desc_block)
        {
            std::lock_guard<std::mutex> lock(allocation_mutex_);
            this->Deallocate(lock, desc_block);
        }
    }

    void D3D12DescriptorAllocator::Deallocate([[maybe_unused]] std::lock_guard<std::mutex>& proof_of_lock, D3D12DescriptorBlock& desc_block)
    {
        assert(desc_block);

        const uint16_t default_page_size = DescriptorPageSize[type_];

        if (desc_block.Size() <= default_page_size)
        {
            for (auto& page : pages_)
            {
                if (&page.page.Heap() == desc_block.Heap())
                {
                    page.stall_list.push_back(
                        {{static_cast<uint16_t>(desc_block.Offset()), static_cast<uint16_t>(desc_block.Offset() + desc_block.Size())},
                            gpu_system_->FenceValue()});
                    return;
                }
            }

            Unreachable("This descriptor block is not allocated by this allocator");
        }
    }

    void D3D12DescriptorAllocator::Reallocate(D3D12DescriptorBlock& desc_block, uint32_t size)
    {
        std::lock_guard<std::mutex> lock(allocation_mutex_);

        if (desc_block)
        {
            this->Deallocate(lock, desc_block);
        }
        this->Allocate(lock, desc_block, size);
    }

    void D3D12DescriptorAllocator::ClearStallPages(uint64_t completed_fence_value)
    {
        std::lock_guard<std::mutex> lock(allocation_mutex_);

        for (auto& page : pages_)
        {
            for (auto stall_iter = page.stall_list.begin(); stall_iter != page.stall_list.end();)
            {
                if (stall_iter->fence_value <= completed_fence_value)
                {
                    const auto free_iter = std::lower_bound(page.free_list.begin(), page.free_list.end(),
                        stall_iter->free_range.first_offset, [](const PageInfo::FreeRange& free_range, uint32_t first_offset) {
                            return free_range.first_offset < first_offset;
                        });
                    if (free_iter == page.free_list.end())
                    {
                        if (page.free_list.empty() || (page.free_list.back().last_offset != stall_iter->free_range.first_offset))
                        {
                            page.free_list.emplace_back(std::move(stall_iter->free_range));
                        }
                        else
                        {
                            page.free_list.back().last_offset = stall_iter->free_range.last_offset;
                        }
                    }
                    else if (free_iter->first_offset != stall_iter->free_range.last_offset)
                    {
                        bool merge_with_prev = false;
                        if (free_iter != page.free_list.begin())
                        {
                            const auto prev_free_iter = std::prev(free_iter);
                            if (prev_free_iter->last_offset == stall_iter->free_range.first_offset)
                            {
                                prev_free_iter->last_offset = stall_iter->free_range.last_offset;
                                merge_with_prev = true;
                            }
                        }

                        if (!merge_with_prev)
                        {
                            page.free_list.emplace(free_iter, std::move(stall_iter->free_range));
                        }
                    }
                    else
                    {
                        free_iter->first_offset = stall_iter->free_range.first_offset;
                        if (free_iter != page.free_list.begin())
                        {
                            const auto prev_free_iter = std::prev(free_iter);
                            if (prev_free_iter->last_offset == free_iter->first_offset)
                            {
                                prev_free_iter->last_offset = free_iter->last_offset;
                                page.free_list.erase(free_iter);
                            }
                        }
                    }

                    stall_iter = page.stall_list.erase(stall_iter);
                }
                else
                {
                    ++stall_iter;
                }
            }
        }
    }

    void D3D12DescriptorAllocator::Clear()
    {
        std::lock_guard<std::mutex> lock(allocation_mutex_);

        pages_.clear();
    }
} // namespace AIHoloImager
