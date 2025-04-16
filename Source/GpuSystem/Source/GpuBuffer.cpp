// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuBuffer.hpp"

#include "Base/ErrorHandling.hpp"
#include "Base/Uuid.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    D3D12_RANGE ToD3D12Range(const GpuRange& range)
    {
        return D3D12_RANGE{range.begin, range.end};
    }


    GpuBuffer::GpuBuffer() noexcept = default;

    GpuBuffer::GpuBuffer(GpuSystem& gpu_system, uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name)
        : resource_(gpu_system, nullptr), heap_type_(ToD3D12HeapType(heap)),
          curr_state_(heap == GpuHeap::ReadBack ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COMMON)
    {
        ID3D12Device* d3d12_device = gpu_system.NativeDevice();

        const D3D12_HEAP_PROPERTIES heap_prop = {heap_type_, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};

        desc_ = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, static_cast<uint64_t>(size), 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ToD3D12ResourceFlags(flags)};
        TIFHR(d3d12_device->CreateCommittedResource(
            &heap_prop, ToD3D12HeapFlags(flags), &desc_, curr_state_, nullptr, UuidOf<ID3D12Resource>(), resource_.Object().PutVoid()));
        if (!name.empty())
        {
            resource_->SetName(std::wstring(std::move(name)).c_str());
        }

        if (EnumHasAny(flags, GpuResourceFlag::Shareable))
        {
            HANDLE shared_handle;
            TIFHR(d3d12_device->CreateSharedHandle(this->NativeBuffer(), nullptr, GENERIC_ALL, nullptr, &shared_handle));
            shared_handle_.reset(shared_handle);
        }
    }

    GpuBuffer::GpuBuffer(GpuSystem& gpu_system, ID3D12Resource* native_resource, GpuResourceState curr_state, std::wstring_view name)
        : resource_(gpu_system, ComPtr<ID3D12Resource>(native_resource, false)), curr_state_(ToD3D12ResourceState(curr_state))
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

    GpuBuffer::~GpuBuffer() = default;

    GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept = default;
    GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept = default;

    GpuBuffer GpuBuffer::Share() const
    {
        GpuBuffer buffer;
        buffer.resource_ = resource_.Share();
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
        return resource_.Object().Get();
    }

    D3D12_GPU_VIRTUAL_ADDRESS GpuBuffer::GpuVirtualAddress() const noexcept
    {
        return resource_->GetGPUVirtualAddress();
    }

    uint32_t GpuBuffer::Size() const noexcept
    {
        return static_cast<uint32_t>(desc_.Width);
    }

    void* GpuBuffer::Map(const GpuRange& read_range)
    {
        void* addr;
        const D3D12_RANGE d3d12_read_range = ToD3D12Range(read_range);
        TIFHR(resource_->Map(0, &d3d12_read_range, &addr));
        return addr;
    }

    void* GpuBuffer::Map()
    {
        void* addr;
        const D3D12_RANGE d3d12_read_range{0, 0};
        TIFHR(resource_->Map(0, (heap_type_ == D3D12_HEAP_TYPE_READBACK) ? nullptr : &d3d12_read_range, &addr));
        return addr;
    }

    void GpuBuffer::Unmap(const GpuRange& write_range)
    {
        const D3D12_RANGE d3d12_write_range = ToD3D12Range(write_range);
        resource_->Unmap(0, (heap_type_ == D3D12_HEAP_TYPE_UPLOAD) ? nullptr : &d3d12_write_range);
    }

    void GpuBuffer::Unmap()
    {
        this->Unmap(GpuRange{0, 0});
    }

    void GpuBuffer::Reset()
    {
        resource_.Reset();
        desc_ = {};
        heap_type_ = {};
        curr_state_ = {};
    }

    void GpuBuffer::Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const
    {
        const D3D12_RESOURCE_STATES d3d12_target_state = ToD3D12ResourceState(target_state);

        D3D12_RESOURCE_BARRIER barrier;
        if (curr_state_ != d3d12_target_state)
        {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = resource_.Object().Get();
            barrier.Transition.StateBefore = curr_state_;
            barrier.Transition.StateAfter = d3d12_target_state;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmd_list.Transition(std::span(&barrier, 1));
        }
        else if ((target_state == GpuResourceState::UnorderedAccess) || (target_state == GpuResourceState::RayTracingAS))
        {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.UAV.pResource = resource_.Object().Get();
            cmd_list.Transition(std::span(&barrier, 1));
        }

        curr_state_ = d3d12_target_state;
    }

    HANDLE GpuBuffer::SharedHandle() const noexcept
    {
        return shared_handle_.get();
    }


    GpuUploadBuffer::GpuUploadBuffer() noexcept = default;

    GpuUploadBuffer::GpuUploadBuffer(GpuSystem& gpu_system, uint32_t size, std::wstring_view name)
        : GpuBuffer(gpu_system, size, GpuHeap::Upload, GpuResourceFlag::None, std::move(name)), mapped_data_(this->Map())
    {
    }

    GpuUploadBuffer::GpuUploadBuffer(GpuSystem& gpu_system, const void* data, uint32_t size, std::wstring_view name)
        : GpuUploadBuffer(gpu_system, size, std::move(name))
    {
        memcpy(MappedData<void>(), data, size);
    }

    GpuUploadBuffer::~GpuUploadBuffer() noexcept
    {
        if (resource_)
        {
            this->Unmap();
        }
    }

    GpuUploadBuffer::GpuUploadBuffer(GpuUploadBuffer&& other) noexcept = default;
    GpuUploadBuffer& GpuUploadBuffer::operator=(GpuUploadBuffer&& other) noexcept = default;

    GpuUploadBuffer GpuUploadBuffer::Share() const
    {
        GpuUploadBuffer buffer;
        buffer.resource_ = resource_.Share();
        buffer.desc_ = desc_;
        buffer.heap_type_ = heap_type_;
        buffer.curr_state_ = curr_state_;
        buffer.mapped_data_ = mapped_data_;
        return buffer;
    }

    void GpuUploadBuffer::Reset()
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


    GpuReadbackBuffer::GpuReadbackBuffer() noexcept = default;

    GpuReadbackBuffer::GpuReadbackBuffer(GpuSystem& gpu_system, uint32_t size, std::wstring_view name)
        : GpuBuffer(gpu_system, size, GpuHeap::ReadBack, GpuResourceFlag::None, std::move(name)), mapped_data_(this->Map())
    {
    }

    GpuReadbackBuffer::GpuReadbackBuffer(GpuSystem& gpu_system, const void* data, uint32_t size, std::wstring_view name)
        : GpuReadbackBuffer(gpu_system, size, std::move(name))
    {
        memcpy(MappedData<void>(), data, size);
    }

    GpuReadbackBuffer::~GpuReadbackBuffer() noexcept
    {
        if (resource_)
        {
            this->Unmap();
        }
    }

    GpuReadbackBuffer::GpuReadbackBuffer(GpuReadbackBuffer&& other) noexcept = default;
    GpuReadbackBuffer& GpuReadbackBuffer::operator=(GpuReadbackBuffer&& other) noexcept = default;

    GpuReadbackBuffer GpuReadbackBuffer::Share() const
    {
        GpuReadbackBuffer buffer;
        buffer.resource_ = resource_.Share();
        buffer.desc_ = desc_;
        buffer.heap_type_ = heap_type_;
        buffer.curr_state_ = curr_state_;
        buffer.mapped_data_ = mapped_data_;
        return buffer;
    }

    void GpuReadbackBuffer::Reset()
    {
        if (resource_)
        {
            this->Unmap();
        }

        GpuBuffer::Reset();
    }

    void* GpuReadbackBuffer::MappedData() noexcept
    {
        return mapped_data_;
    }

    const void* GpuReadbackBuffer::MappedData() const noexcept
    {
        return mapped_data_;
    }
} // namespace AIHoloImager
