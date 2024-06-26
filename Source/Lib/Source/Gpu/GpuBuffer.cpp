// Copyright (c) 2024 Minmin Gong
//

#include "GpuBuffer.hpp"

#include "GpuCommandList.hpp"
#include "GpuSystem.hpp"
#include "Util/ErrorHandling.hpp"
#include "Util/Uuid.hpp"

namespace AIHoloImager
{
    GpuBuffer::GpuBuffer() noexcept = default;

    GpuBuffer::GpuBuffer(GpuSystem& gpu_system, uint32_t size, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_FLAGS flags,
        D3D12_RESOURCE_STATES init_state, std::wstring_view name)
        : heap_type_(heap_type), curr_state_(init_state)
    {
        const D3D12_HEAP_PROPERTIES heap_prop = {heap_type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};

        desc_ = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, static_cast<uint64_t>(size), 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags};
        TIFHR(gpu_system.NativeDevice()->CreateCommittedResource(
            &heap_prop, D3D12_HEAP_FLAG_NONE, &desc_, init_state, nullptr, UuidOf<ID3D12Resource>(), resource_.PutVoid()));
        if (!name.empty())
        {
            resource_->SetName(std::wstring(std::move(name)).c_str());
        }
    }

    GpuBuffer::GpuBuffer(ID3D12Resource* native_resource, D3D12_RESOURCE_STATES curr_state, std::wstring_view name)
        : resource_(native_resource, false), curr_state_(curr_state)
    {
        if (resource_)
        {
            D3D12_HEAP_PROPERTIES heap_prop;
            resource_->GetHeapProperties(&heap_prop, nullptr);
            heap_type_ = heap_prop.Type;

            desc_ = resource_->GetDesc();
            if (!name.empty())
            {
                resource_->SetName(std::wstring(std::move(name)).c_str());
            }
        }
    }

    GpuBuffer::~GpuBuffer() noexcept = default;

    GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept = default;
    GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept = default;

    GpuBuffer GpuBuffer::Share() const
    {
        GpuBuffer buffer;
        buffer.resource_ = resource_;
        buffer.desc_ = desc_;
        buffer.heap_type_ = heap_type_;
        buffer.curr_state_ = curr_state_;
        return buffer;
    }

    GpuBuffer::operator bool() const noexcept
    {
        return resource_ ? true : false;
    }

    ID3D12Resource* GpuBuffer::NativeBuffer() const noexcept
    {
        return resource_.Get();
    }

    D3D12_GPU_VIRTUAL_ADDRESS GpuBuffer::GpuVirtualAddress() const noexcept
    {
        return resource_->GetGPUVirtualAddress();
    }

    uint32_t GpuBuffer::Size() const noexcept
    {
        return static_cast<uint32_t>(desc_.Width);
    }

    void* GpuBuffer::Map(const D3D12_RANGE& read_range)
    {
        void* addr;
        TIFHR(resource_->Map(0, &read_range, &addr));
        return addr;
    }

    void* GpuBuffer::Map()
    {
        void* addr;
        const D3D12_RANGE read_range{0, 0};
        TIFHR(resource_->Map(0, (heap_type_ == D3D12_HEAP_TYPE_READBACK) ? nullptr : &read_range, &addr));
        return addr;
    }

    void GpuBuffer::Unmap(const D3D12_RANGE& write_range)
    {
        resource_->Unmap(0, (heap_type_ == D3D12_HEAP_TYPE_UPLOAD) ? nullptr : &write_range);
    }

    void GpuBuffer::Unmap()
    {
        this->Unmap(D3D12_RANGE{0, 0});
    }

    void GpuBuffer::Reset() noexcept
    {
        resource_ = nullptr;
        desc_ = {};
        heap_type_ = {};
        curr_state_ = {};
    }

    D3D12_RESOURCE_STATES GpuBuffer::State() const noexcept
    {
        return curr_state_;
    }

    void GpuBuffer::Transition(GpuCommandList& cmd_list, D3D12_RESOURCE_STATES target_state)
    {
        D3D12_RESOURCE_BARRIER barrier;
        if (curr_state_ != target_state)
        {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = resource_.Get();
            barrier.Transition.StateBefore = curr_state_;
            barrier.Transition.StateAfter = target_state;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmd_list.Transition(std::span(&barrier, 1));
        }
        else if ((target_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
                 (target_state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE))
        {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.UAV.pResource = resource_.Get();
            cmd_list.Transition(std::span(&barrier, 1));
        }

        curr_state_ = target_state;
    }


    GpuUploadBuffer::GpuUploadBuffer() noexcept = default;

    GpuUploadBuffer::GpuUploadBuffer(GpuSystem& gpu_system, uint32_t size, std::wstring_view name)
        : GpuBuffer(gpu_system, size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, std::move(name)),
          mapped_data_(this->Map())
    {
    }

    GpuUploadBuffer::GpuUploadBuffer(GpuSystem& gpu_system, const void* data, uint32_t size, std::wstring_view name)
        : GpuUploadBuffer(gpu_system, size, std::move(name))
    {
        memcpy(MappedData<void>(), data, size);
    }

    GpuUploadBuffer::~GpuUploadBuffer() noexcept
    {
        this->Reset();
    }

    GpuUploadBuffer::GpuUploadBuffer(GpuUploadBuffer&& other) noexcept = default;
    GpuUploadBuffer& GpuUploadBuffer::operator=(GpuUploadBuffer&& other) noexcept = default;

    GpuUploadBuffer GpuUploadBuffer::Share() const
    {
        GpuUploadBuffer buffer;
        buffer.resource_ = resource_;
        buffer.desc_ = desc_;
        buffer.heap_type_ = heap_type_;
        buffer.curr_state_ = curr_state_;
        buffer.mapped_data_ = mapped_data_;
        return buffer;
    }

    void GpuUploadBuffer::Reset() noexcept
    {
        if (resource_)
        {
            this->Unmap();
        }

        GpuBuffer::Reset();
    }

    void* GpuUploadBuffer::MappedData() noexcept
    {
        return mapped_data_;
    }
} // namespace AIHoloImager
