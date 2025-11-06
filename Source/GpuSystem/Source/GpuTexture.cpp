// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuTexture.hpp"

#include <cassert>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSystem.hpp"

#include "Internal/GpuSystemInternal.hpp"
#include "Internal/GpuTextureInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    void DecomposeSubResource(uint32_t sub_resource, uint32_t num_mip_levels, uint32_t array_size, uint32_t& mip_slice,
        uint32_t& array_slice, uint32_t& plane_slice) noexcept
    {
        const uint32_t plane_array = sub_resource / num_mip_levels;
        mip_slice = sub_resource - plane_array * num_mip_levels;
        plane_slice = plane_array / array_size;
        array_slice = plane_array - plane_slice * array_size;
    }

    uint32_t CalcSubResource(
        uint32_t mip_slice, uint32_t array_slice, uint32_t plane_slice, uint32_t num_mip_levels, uint32_t array_size) noexcept
    {
        return (plane_slice * array_size + array_slice) * num_mip_levels + mip_slice;
    }


    EMPTY_IMP(GpuTexture)
    IMP_INTERNAL2(GpuTexture, GpuResource)

    GpuTexture::GpuTexture() noexcept = default;

    GpuTexture::GpuTexture(GpuSystem& gpu_system, GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
        uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
        : impl_(static_cast<Impl*>(gpu_system.Internal()
                                       .CreateTexture(type, width, height, depth, array_size, mip_levels, format, flags, std::move(name))
                                       .release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuTextureInternal));
    }

    GpuTexture::GpuTexture(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name) noexcept
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateTexture(native_resource, curr_state, std::move(name)).release()))
    {
    }

    GpuTexture::~GpuTexture() = default;

    GpuTexture::GpuTexture(GpuTexture&& other) noexcept = default;
    GpuTexture& GpuTexture::operator=(GpuTexture&& other) noexcept = default;

    void GpuTexture::Name(std::wstring_view name)
    {
        assert(impl_);
        return impl_->Name(std::move(name));
    }

    void* GpuTexture::NativeResource() const noexcept
    {
        return impl_ ? impl_->NativeResource() : nullptr;
    }

    void* GpuTexture::NativeTexture() const noexcept
    {
        return impl_ ? impl_->NativeTexture() : nullptr;
    }

    GpuTexture::operator bool() const noexcept
    {
        return this->NativeTexture() != nullptr;
    }

    void* GpuTexture::SharedHandle() const noexcept
    {
        return impl_ ? impl_->SharedHandle() : nullptr;
    }

    GpuResourceType GpuTexture::Type() const noexcept
    {
        assert(impl_);
        return impl_->Type();
    }

    uint32_t GpuTexture::AllocationSize() const noexcept
    {
        assert(impl_);
        return impl_->AllocationSize();
    }

    uint32_t GpuTexture::Width(uint32_t mip) const noexcept
    {
        return impl_ ? impl_->Width(mip) : 0;
    }

    uint32_t GpuTexture::Height(uint32_t mip) const noexcept
    {
        return impl_ ? impl_->Height(mip) : 0;
    }

    uint32_t GpuTexture::Depth(uint32_t mip) const noexcept
    {
        return impl_ ? impl_->Depth(mip) : 0;
    }

    uint32_t GpuTexture::ArraySize() const noexcept
    {
        return impl_ ? impl_->ArraySize() : 0;
    }

    uint32_t GpuTexture::MipLevels() const noexcept
    {
        return impl_ ? impl_->MipLevels() : 0;
    }

    uint32_t GpuTexture::Planes() const noexcept
    {
        return impl_ ? impl_->Planes() : 0;
    }

    GpuFormat GpuTexture::Format() const noexcept
    {
        return impl_ ? impl_->Format() : GpuFormat::Unknown;
    }

    GpuResourceFlag GpuTexture::Flags() const noexcept
    {
        return impl_ ? impl_->Flags() : GpuResourceFlag::None;
    }

    void GpuTexture::Reset()
    {
        assert(impl_);
        impl_->Reset();
    }

    void GpuTexture::Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const
    {
        assert(impl_);
        impl_->Transition(cmd_list, sub_resource, target_state);
    }

    void GpuTexture::Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const
    {
        assert(impl_);
        impl_->Transition(cmd_list, target_state);
    }


    GpuTexture2D::GpuTexture2D() noexcept = default;

    GpuTexture2D::GpuTexture2D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t mip_levels, GpuFormat format,
        GpuResourceFlag flags, std::wstring_view name)
        : GpuTexture(gpu_system, GpuResourceType::Texture2D, width, height, 1, 1, mip_levels, format, flags, std::move(name))
    {
    }

    GpuTexture2D::GpuTexture2D(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name) noexcept
        : GpuTexture(gpu_system, native_resource, curr_state, std::move(name))
    {
    }

    GpuTexture2D::~GpuTexture2D() = default;

    GpuTexture2D::GpuTexture2D(GpuTexture2D&& other) noexcept = default;
    GpuTexture2D& GpuTexture2D::operator=(GpuTexture2D&& other) noexcept = default;


    GpuTexture2DArray::GpuTexture2DArray() noexcept = default;

    GpuTexture2DArray::GpuTexture2DArray(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t array_size, uint32_t mip_levels,
        GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
        : GpuTexture(gpu_system, GpuResourceType::Texture2DArray, width, height, 1, array_size, mip_levels, format, flags, std::move(name))
    {
    }

    GpuTexture2DArray::GpuTexture2DArray(
        GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name) noexcept
        : GpuTexture(gpu_system, native_resource, curr_state, std::move(name))
    {
    }

    GpuTexture2DArray::~GpuTexture2DArray() = default;

    GpuTexture2DArray::GpuTexture2DArray(GpuTexture2DArray&& other) noexcept = default;
    GpuTexture2DArray& GpuTexture2DArray::operator=(GpuTexture2DArray&& other) noexcept = default;


    GpuTexture3D::GpuTexture3D() noexcept = default;

    GpuTexture3D::GpuTexture3D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t depth, uint32_t mip_levels,
        GpuFormat format, GpuResourceFlag flags, std::wstring_view name)
        : GpuTexture(gpu_system, GpuResourceType::Texture3D, width, height, depth, 1, mip_levels, format, flags, std::move(name))
    {
    }

    GpuTexture3D::GpuTexture3D(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name) noexcept
        : GpuTexture(gpu_system, native_resource, curr_state, std::move(name))
    {
    }

    GpuTexture3D::~GpuTexture3D() = default;

    GpuTexture3D::GpuTexture3D(GpuTexture3D&& other) noexcept = default;
    GpuTexture3D& GpuTexture3D::operator=(GpuTexture3D&& other) noexcept = default;
} // namespace AIHoloImager
