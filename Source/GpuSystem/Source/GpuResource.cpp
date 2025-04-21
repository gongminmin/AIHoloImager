// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuResource.hpp"

#include "Base/ErrorHandling.hpp"
#include "Gpu/GpuSystem.hpp"

namespace AIHoloImager
{
    D3D12_HEAP_TYPE ToD3D12HeapType(GpuHeap heap)
    {
        switch (heap)
        {
        case GpuHeap::Default:
            return D3D12_HEAP_TYPE_DEFAULT;
        case GpuHeap::Upload:
            return D3D12_HEAP_TYPE_UPLOAD;
        case GpuHeap::ReadBack:
            return D3D12_HEAP_TYPE_READBACK;

        default:
            Unreachable("Invalid heap");
        }
    }

    D3D12_RESOURCE_FLAGS ToD3D12ResourceFlags(GpuResourceFlag flags) noexcept
    {
        D3D12_RESOURCE_FLAGS d3d12_flag = D3D12_RESOURCE_FLAG_NONE;
        if (EnumHasAny(flags, GpuResourceFlag::RenderTarget))
        {
            d3d12_flag |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }
        if (EnumHasAny(flags, GpuResourceFlag::DepthStencil))
        {
            d3d12_flag |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        }
        if (EnumHasAny(flags, GpuResourceFlag::UnorderedAccess))
        {
            d3d12_flag |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        return d3d12_flag;
    }

    D3D12_HEAP_FLAGS ToD3D12HeapFlags(GpuResourceFlag flags) noexcept
    {
        D3D12_HEAP_FLAGS heap_flags = D3D12_HEAP_FLAG_NONE;
        if (EnumHasAny(flags, GpuResourceFlag::Shareable))
        {
            heap_flags |= D3D12_HEAP_FLAG_SHARED;
        }
        return heap_flags;
    }

    D3D12_RESOURCE_STATES ToD3D12ResourceState(GpuResourceState state)
    {
        switch (state)
        {
        case GpuResourceState::Common:
            return D3D12_RESOURCE_STATE_COMMON;

        case GpuResourceState::ColorWrite:
            return D3D12_RESOURCE_STATE_RENDER_TARGET;
        case GpuResourceState::DepthWrite:
            return D3D12_RESOURCE_STATE_DEPTH_WRITE;

        case GpuResourceState::UnorderedAccess:
            return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        case GpuResourceState::CopySrc:
            return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case GpuResourceState::CopyDst:
            return D3D12_RESOURCE_STATE_COPY_DEST;

        default:
            Unreachable("Invalid resource state");
        }
    }


    GpuResource::GpuResource() = default;

    GpuResource::GpuResource(GpuSystem& gpu_system) : resource_(gpu_system, nullptr)
    {
    }

    GpuResource::GpuResource(GpuSystem& gpu_system, ID3D12Resource* native_resource)
        : resource_(gpu_system, ComPtr<ID3D12Resource>(native_resource, false))
    {
    }

    GpuResource::~GpuResource() = default;

    GpuResource::GpuResource(GpuResource&& other) noexcept = default;
    GpuResource& GpuResource::operator=(GpuResource&& other) noexcept = default;

    void GpuResource::Name(std::wstring_view name)
    {
        resource_->SetName(name.empty() ? L"" : std::wstring(std::move(name)).c_str());
    }

    ID3D12Resource* GpuResource::NativeResource() const noexcept
    {
        return resource_.Object().Get();
    }

    GpuResource::operator bool() const noexcept
    {
        return resource_ ? true : false;
    }

    void GpuResource::Reset()
    {
        resource_.Reset();
        desc_ = {};
    }

    void GpuResource::CreateSharedHandle(GpuSystem& gpu_system, GpuResourceFlag flags)
    {
        if (EnumHasAny(flags, GpuResourceFlag::Shareable))
        {
            ID3D12Device* d3d12_device = gpu_system.NativeDevice();

            HANDLE shared_handle;
            TIFHR(d3d12_device->CreateSharedHandle(this->NativeResource(), nullptr, GENERIC_ALL, nullptr, &shared_handle));
            shared_handle_.reset(shared_handle);
        }
    }

    HANDLE GpuResource::SharedHandle() const noexcept
    {
        return shared_handle_.get();
    }
} // namespace AIHoloImager
