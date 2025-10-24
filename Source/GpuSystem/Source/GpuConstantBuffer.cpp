// Copyright (c) 2025 Minmin Gong
//

#include "Gpu/GpuConstantBuffer.hpp"

namespace AIHoloImager
{
    GpuConstantBuffer::GpuConstantBuffer() noexcept = default;
    GpuConstantBuffer::GpuConstantBuffer(GpuSystem& gpu_system, uint32_t size, std::wstring_view name)
        : gpu_system_(&gpu_system), mem_block_(gpu_system.AllocUploadMemBlock(size, gpu_system.ConstantDataAlignment())),
          name_(std::move(name))
    {
    }

    GpuConstantBuffer::~GpuConstantBuffer()
    {
        if ((gpu_system_ != nullptr) && mem_block_)
        {
            gpu_system_->DeallocUploadMemBlock(std::move(mem_block_));
        }
    }

    GpuConstantBuffer::GpuConstantBuffer(GpuConstantBuffer&& other) noexcept = default;
    GpuConstantBuffer& GpuConstantBuffer::operator=(GpuConstantBuffer&& other) noexcept = default;

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
        return mem_block_ ? mem_block_.Buffer()->NativeBuffer() : nullptr;
    }

    GpuVirtualAddressType GpuConstantBuffer::GpuVirtualAddress() const noexcept
    {
        return mem_block_.GpuAddress();
    }
} // namespace AIHoloImager
