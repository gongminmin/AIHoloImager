// Copyright (c) 2025 Minmin Gong
//

#include "Gpu/GpuConstantBuffer.hpp"

namespace AIHoloImager
{
    GpuConstantBuffer::GpuConstantBuffer() noexcept = default;
    GpuConstantBuffer::GpuConstantBuffer(GpuSystem& gpu_system, uint32_t size, std::wstring_view name)
        : gpu_system_(&gpu_system), mem_block_(gpu_system.AllocUploadMemBlock(size, GpuMemoryAllocator::ConstantDataAlignment)),
          name_(std::move(name))
    {
    }
    GpuConstantBuffer::GpuConstantBuffer(GpuConstantBuffer&& other) noexcept
        : gpu_system_(other.gpu_system_), mem_block_(std::move(other.mem_block_)), name_(std::move(name_))
    {
    }

    GpuConstantBuffer::~GpuConstantBuffer()
    {
        if ((gpu_system_ != nullptr) && mem_block_)
        {
            gpu_system_->DeallocUploadMemBlock(std::move(mem_block_));
        }
    }

    GpuConstantBuffer& GpuConstantBuffer::operator=(GpuConstantBuffer&& other) noexcept
    {
        if (this != &other)
        {
            gpu_system_ = std::exchange(other.gpu_system_, {});
            mem_block_ = std::move(other.mem_block_);
            name_ = std::move(name_);
        }
        return *this;
    }

    GpuConstantBuffer::operator bool() const noexcept
    {
        return mem_block_ ? true : false;
    }

    const GpuMemoryBlock& GpuConstantBuffer::MemBlock() const noexcept
    {
        return mem_block_;
    }

    void* GpuConstantBuffer::NativeResource() const noexcept
    {
        return mem_block_.NativeBuffer();
    }

    GpuVirtualAddressType GpuConstantBuffer::GpuVirtualAddress() const noexcept
    {
        return mem_block_.GpuAddress();
    }
} // namespace AIHoloImager
