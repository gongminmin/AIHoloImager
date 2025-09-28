// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuResource.hpp"

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Base/ErrorHandling.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Gpu/D3D12/D3D12Traits.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuUtil.hpp"

#include "D3D12/D3D12Conversion.hpp"

namespace AIHoloImager
{
    class GpuResource::Impl
    {
    public:
        Impl() = default;

        explicit Impl(GpuSystem& gpu_system) : resource_(gpu_system, nullptr)
        {
        }
        Impl(GpuSystem& gpu_system, void* native_resource, std::wstring_view name)
            : resource_(gpu_system, ComPtr<ID3D12Resource>(reinterpret_cast<ID3D12Resource*>(native_resource), false))
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

        ~Impl() = default;

        Impl(Impl&& other) noexcept = default;
        Impl& operator=(Impl&& other) noexcept = default;

        void Name(std::wstring_view name)
        {
            resource_->SetName(name.empty() ? L"" : std::wstring(std::move(name)).c_str());
        }

        void* NativeResource() const noexcept
        {
            return resource_.Object().Get();
        }
        template <typename Traits>
        typename Traits::ResourceType NativeResource() const noexcept
        {
            return reinterpret_cast<typename Traits::ResourceType>(this->NativeResource());
        }

        explicit operator bool() const noexcept
        {
            return resource_ ? true : false;
        }

        void Reset()
        {
            resource_.Reset();
            desc_ = {};
        }

        void CreateResource(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size, uint32_t mip_levels,
            GpuFormat format, GpuHeap heap, GpuResourceFlag flags, GpuResourceState init_state, std::wstring_view name)
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

            desc_ = {ToD3D12ResourceDimension(type), 0, static_cast<uint64_t>(width), height, depth_or_array_size,
                static_cast<uint16_t>(mip_levels), ToDxgiFormat(format), {1, 0}, layout, ToD3D12ResourceFlags(flags)};

            auto& gpu_system = *resource_.GpuSys();
            ID3D12Device* d3d12_device = gpu_system.NativeDevice();

            const D3D12_HEAP_PROPERTIES heap_prop = {
                ToD3D12HeapType(heap), D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};

            TIFHR(d3d12_device->CreateCommittedResource(&heap_prop, ToD3D12HeapFlags(flags), &desc_, ToD3D12ResourceState(init_state),
                nullptr, UuidOf<ID3D12Resource>(), resource_.Object().PutVoid()));
            this->Name(std::move(name));

            if (EnumHasAny(flags, GpuResourceFlag::Shareable))
            {
                HANDLE shared_handle;
                TIFHR(d3d12_device->CreateSharedHandle(this->NativeResource<D3D12Traits>(), nullptr, GENERIC_ALL, nullptr, &shared_handle));
                shared_handle_.reset(shared_handle);
            }
        }

        void* SharedHandle() const noexcept
        {
            return shared_handle_.get();
        }

        GpuResourceType Type() const noexcept
        {
            return type_;
        }

        uint32_t Width() const noexcept
        {
            return static_cast<uint32_t>(desc_.Width);
        }

        uint32_t Height() const noexcept
        {
            return desc_.Height;
        }

        uint32_t Depth() const noexcept
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

        uint32_t ArraySize() const noexcept
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

        uint32_t MipLevels() const noexcept
        {
            return desc_.MipLevels;
        }

        GpuFormat Format() const noexcept
        {
            return FromDxgiFormat(desc_.Format);
        }

        GpuResourceFlag Flags() const noexcept
        {
            return FromD3D12ResourceFlags(desc_.Flags);
        }

    private:
        GpuRecyclableObject<ComPtr<ID3D12Resource>> resource_;
        GpuResourceType type_;
        D3D12_RESOURCE_DESC desc_{};
        Win32UniqueHandle shared_handle_;
    };


    GpuResource::GpuResource() noexcept : impl_(std::make_unique<Impl>())
    {
    }

    GpuResource::GpuResource(GpuSystem& gpu_system) : impl_(std::make_unique<Impl>(gpu_system))
    {
    }

    GpuResource::GpuResource(GpuSystem& gpu_system, void* native_resource, std::wstring_view name)
        : impl_(std::make_unique<Impl>(gpu_system, native_resource, std::move(name)))
    {
    }

    GpuResource::~GpuResource() = default;

    GpuResource::GpuResource(GpuResource&& other) noexcept = default;
    GpuResource& GpuResource::operator=(GpuResource&& other) noexcept = default;

    void GpuResource::Name(std::wstring_view name)
    {
        assert(impl_);
        impl_->Name(std::move(name));
    }

    void* GpuResource::NativeResource() const noexcept
    {
        assert(impl_);
        return impl_->NativeResource();
    }

    GpuResource::operator bool() const noexcept
    {
        return impl_ && impl_->operator bool();
    }

    void GpuResource::Reset()
    {
        assert(impl_);
        impl_->Reset();
    }

    void GpuResource::CreateResource(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size,
        uint32_t mip_levels, GpuFormat format, GpuHeap heap, GpuResourceFlag flags, GpuResourceState init_state, std::wstring_view name)
    {
        assert(impl_);
        impl_->CreateResource(type, width, height, depth, array_size, mip_levels, format, heap, flags, init_state, std::move(name));
    }

    void* GpuResource::SharedHandle() const noexcept
    {
        assert(impl_);
        return impl_->SharedHandle();
    }

    GpuResourceType GpuResource::Type() const noexcept
    {
        assert(impl_);
        return impl_->Type();
    }

    uint32_t GpuResource::Width() const noexcept
    {
        assert(impl_);
        return impl_->Width();
    }

    uint32_t GpuResource::Height() const noexcept
    {
        assert(impl_);
        return impl_->Height();
    }

    uint32_t GpuResource::Depth() const noexcept
    {
        assert(impl_);
        return impl_->Depth();
    }

    uint32_t GpuResource::ArraySize() const noexcept
    {
        assert(impl_);
        return impl_->ArraySize();
    }

    uint32_t GpuResource::MipLevels() const noexcept
    {
        assert(impl_);
        return impl_->MipLevels();
    }

    GpuFormat GpuResource::Format() const noexcept
    {
        assert(impl_);
        return impl_->Format();
    }

    GpuResourceFlag GpuResource::Flags() const noexcept
    {
        assert(impl_);
        return impl_->Flags();
    }
} // namespace AIHoloImager
