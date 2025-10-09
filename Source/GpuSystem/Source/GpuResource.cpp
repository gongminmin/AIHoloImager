// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuResource.hpp"

#include "Gpu/GpuSystem.hpp"

#include "Internal/GpuResourceInternal.hpp"
#include "Internal/GpuSystemInternalFactory.hpp"

namespace AIHoloImager
{
    class GpuResource::Impl : public GpuResourceInternal
    {
    };

    GpuResource::GpuResource() noexcept = default;
    GpuResource::GpuResource(GpuSystem& gpu_system) : impl_(static_cast<Impl*>(gpu_system.InternalFactory().CreateGpuResource().release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuResourceInternal));
    }
    GpuResource::GpuResource(GpuSystem& gpu_system, void* native_resource, std::wstring_view name)
        : impl_(static_cast<Impl*>(gpu_system.InternalFactory().CreateGpuResource(native_resource, std::move(name)).release()))
    {
    }

    GpuResource::~GpuResource() = default;

    GpuResource::GpuResource(GpuResource&& other) noexcept = default;
    GpuResource& GpuResource::operator=(GpuResource&& other) noexcept = default;

    void GpuResource::Name(std::wstring_view name)
    {
        if (impl_)
        {
            impl_->Name(std::move(name));
        }
    }

    void* GpuResource::NativeResource() const noexcept
    {
        return impl_ ? impl_->NativeResource() : nullptr;
    }

    GpuResource::operator bool() const noexcept
    {
        return impl_ && (impl_->NativeResource() != nullptr);
    }

    void GpuResource::Reset()
    {
        if (impl_)
        {
            impl_->Reset();
        }
    }

    void GpuResource::CreateResource(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size,
        uint32_t mip_levels, GpuFormat format, GpuHeap heap, GpuResourceFlag flags, GpuResourceState init_state, std::wstring_view name)
    {
        assert(impl_);
        impl_->CreateResource(type, width, height, depth, array_size, mip_levels, format, heap, flags, init_state, std::move(name));
    }

    void* GpuResource::SharedHandle() const noexcept
    {
        return impl_ ? impl_->SharedHandle() : nullptr;
    }

    GpuResourceType GpuResource::Type() const noexcept
    {
        assert(impl_);
        return impl_->Type();
    }

    uint32_t GpuResource::Width() const noexcept
    {
        return impl_ ? impl_->Width() : 0;
    }

    uint32_t GpuResource::Height() const noexcept
    {
        return impl_ ? impl_->Height() : 0;
    }

    uint32_t GpuResource::Depth() const noexcept
    {
        return impl_ ? impl_->Depth() : 0;
    }

    uint32_t GpuResource::ArraySize() const noexcept
    {
        return impl_ ? impl_->ArraySize() : 0;
    }

    uint32_t GpuResource::MipLevels() const noexcept
    {
        return impl_ ? impl_->MipLevels() : 0;
    }

    GpuFormat GpuResource::Format() const noexcept
    {
        return impl_ ? impl_->Format() : GpuFormat::Unknown;
    }

    GpuResourceFlag GpuResource::Flags() const noexcept
    {
        return impl_ ? impl_->Flags() : GpuResourceFlag::None;
    }
} // namespace AIHoloImager
