// Copyright (c) 2025 Minmin Gong
//

#include "D3D12Resource.hpp"

#include "D3D12Buffer.hpp"
#include "D3D12Conversion.hpp"
#include "D3D12System.hpp"
#include "D3D12Texture.hpp"
#include "D3D12Util.hpp"

namespace AIHoloImager
{
    D3D12Resource::D3D12Resource(GpuSystem& gpu_system) : resource_(D3D12Imp(gpu_system), nullptr)
    {
    }
    D3D12Resource::D3D12Resource(GpuSystem& gpu_system, void* native_resource, std::string_view name)
        : resource_(D3D12Imp(gpu_system), ComPtr<ID3D12Resource>(reinterpret_cast<ID3D12Resource*>(native_resource), false))
    {
        if (resource_)
        {
            desc_ = resource_->GetDesc();
            this->Name(std::move(name));

            switch (desc_.Dimension)
            {
            case D3D12_RESOURCE_DIMENSION_BUFFER:
                type_ = GpuResourceType::Buffer;
                break;
            case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
                if (desc_.DepthOrArraySize > 1)
                {
                    type_ = GpuResourceType::Texture2DArray;
                }
                else
                {
                    type_ = GpuResourceType::Texture2D;
                }
                break;
            case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
                type_ = GpuResourceType::Texture3D;
                break;
            default:
                Unreachable("Invalid resource dimension");
            }
        }
    }

    D3D12Resource::~D3D12Resource() = default;

    D3D12Resource::D3D12Resource(D3D12Resource&& other) noexcept = default;

    D3D12Resource& D3D12Resource::operator=(D3D12Resource&& other) noexcept = default;

    void D3D12Resource::Name(std::string_view name)
    {
        SetName(*resource_.Object(), std::move(name));
    }

    ID3D12Resource* D3D12Resource::Resource() const noexcept
    {
        return resource_.Object().Get();
    }

    void D3D12Resource::Reset()
    {
        resource_.Reset();
        desc_ = {};
    }

    void D3D12Resource::CreateResource(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size,
        uint32_t mip_levels, GpuFormat format, GpuHeap heap, GpuResourceFlag flags, GpuResourceState init_state, std::string_view name)
    {
        type_ = type;

        uint16_t depth_or_array_size;
        D3D12_TEXTURE_LAYOUT layout;
        switch (type)
        {
        case GpuResourceType::Buffer:
            assert((height == 1) && (depth == 1));
            assert(array_size == 1);
            assert(mip_levels == 1);
            depth_or_array_size = 1;
            layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            break;

        case GpuResourceType::Texture2D:
            assert((width > 0) && (height > 0) && (depth == 1));
            assert(array_size == 1);
            assert(mip_levels >= 1);
            depth_or_array_size = 1;
            layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            break;

        case GpuResourceType::Texture2DArray:
            assert((width > 0) && (height > 0) && (depth == 1));
            assert(array_size >= 1);
            assert(mip_levels >= 1);
            depth_or_array_size = static_cast<uint16_t>(array_size);
            layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            break;

        case GpuResourceType::Texture3D:
            assert((width > 0) && (height > 0) && (depth > 0));
            assert(array_size == 1);
            assert(mip_levels >= 1);
            depth_or_array_size = static_cast<uint16_t>(depth);
            layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            break;

        default:
            Unreachable("Invalid resource type");
        }

        desc_ = {
            .Dimension = ToD3D12ResourceDimension(type),
            .Width = static_cast<uint64_t>(width),
            .Height = height,
            .DepthOrArraySize = depth_or_array_size,
            .MipLevels = static_cast<uint16_t>(mip_levels),
            .Format = ToDxgiFormat(format),
            .SampleDesc{
                .Count = 1,
                .Quality = 0,
            },
            .Layout = layout,
            .Flags = ToD3D12ResourceFlags(flags),
        };

        auto& d3d12_system = *resource_.D3D12Sys();
        ID3D12Device* d3d12_device = d3d12_system.Device();

        const D3D12_HEAP_PROPERTIES heap_prop{
            .Type = ToD3D12HeapType(heap),
            .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
            .CreationNodeMask = 1,
            .VisibleNodeMask = 1,
        };

        TIFHR(d3d12_device->CreateCommittedResource(&heap_prop, ToD3D12HeapFlags(flags), &desc_, ToD3D12ResourceState(init_state), nullptr,
            UuidOf<ID3D12Resource>(), resource_.Object().PutVoid()));
        this->Name(std::move(name));

        if (EnumHasAny(flags, GpuResourceFlag::Shareable))
        {
            HANDLE shared_handle;
            TIFHR(d3d12_device->CreateSharedHandle(resource_.Object().Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle));
            shared_handle_.reset(shared_handle);
        }
    }

    void* D3D12Resource::SharedHandle() const noexcept
    {
        return shared_handle_.get();
    }

    GpuResourceType D3D12Resource::Type() const noexcept
    {
        return type_;
    }

    uint32_t D3D12Resource::AllocationSize() const noexcept
    {
        auto& d3d12_system = *resource_.D3D12Sys();
        ID3D12Device* d3d12_device = d3d12_system.Device();
        const auto alloc_info = d3d12_device->GetResourceAllocationInfo(0, 1, &desc_);
        return static_cast<uint32_t>(alloc_info.SizeInBytes);
    }

    uint32_t D3D12Resource::Width() const noexcept
    {
        return static_cast<uint32_t>(desc_.Width);
    }

    uint32_t D3D12Resource::Height() const noexcept
    {
        return desc_.Height;
    }

    uint32_t D3D12Resource::Depth() const noexcept
    {
        if (type_ == GpuResourceType::Texture3D)
        {
            return desc_.DepthOrArraySize;
        }
        else
        {
            return 1;
        }
    }

    uint32_t D3D12Resource::ArraySize() const noexcept
    {
        if (type_ == GpuResourceType::Texture2DArray)
        {
            return desc_.DepthOrArraySize;
        }
        else
        {
            return 1;
        }
    }

    uint32_t D3D12Resource::MipLevels() const noexcept
    {
        return desc_.MipLevels;
    }

    GpuFormat D3D12Resource::Format() const noexcept
    {
        return FromDxgiFormat(desc_.Format);
    }

    GpuResourceFlag D3D12Resource::Flags() const noexcept
    {
        return FromD3D12ResourceFlags(desc_.Flags);
    }

    D3D12Resource& D3D12Imp(GpuResource& resource)
    {
        if (resource.Type() == GpuResourceType::Buffer)
        {
            return static_cast<D3D12Buffer&>(resource.Internal());
        }
        else
        {
            return static_cast<D3D12Texture&>(resource.Internal());
        }
    }

    const D3D12Resource& D3D12Imp(const GpuResource& resource)
    {
        return D3D12Imp(const_cast<GpuResource&>(resource));
    }
} // namespace AIHoloImager
