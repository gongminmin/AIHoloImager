// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuResourceViews.hpp"

#include <cassert>

#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"

#include "Internal/GpuResourceViewsInternal.hpp"
#include "Internal/GpuSystemInternal.hpp"

namespace AIHoloImager
{
    class GpuShaderResourceView::Impl : public GpuShaderResourceViewInternal
    {
    };

    GpuShaderResourceView::GpuShaderResourceView() noexcept = default;

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture)
        : GpuShaderResourceView(gpu_system, texture, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, GpuFormat format)
        : GpuShaderResourceView(gpu_system, texture, ~0U, format)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource)
        : GpuShaderResourceView(gpu_system, texture, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateShaderResourceView(texture, sub_resource, format).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuShaderResourceViewInternal));
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array)
        : GpuShaderResourceView(gpu_system, texture_array, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, GpuFormat format)
        : GpuShaderResourceView(gpu_system, texture_array, ~0U, format)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource)
        : GpuShaderResourceView(gpu_system, texture_array, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateShaderResourceView(texture_array, sub_resource, format).release()))
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture)
        : GpuShaderResourceView(gpu_system, texture, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, GpuFormat format)
        : GpuShaderResourceView(gpu_system, texture, ~0U, format)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource)
        : GpuShaderResourceView(gpu_system, texture, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateShaderResourceView(texture, sub_resource, format).release()))
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuBuffer& buffer, GpuFormat format)
        : GpuShaderResourceView(gpu_system, buffer, 0, buffer.Size() / FormatSize(format), format)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateShaderResourceView(buffer, first_element, num_elements, format).release()))
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t element_size)
        : GpuShaderResourceView(gpu_system, buffer, 0, buffer.Size() / element_size, element_size)
    {
    }

    GpuShaderResourceView::GpuShaderResourceView(
        GpuSystem& gpu_system, const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size)
        : impl_(static_cast<Impl*>(
              gpu_system.Internal().CreateShaderResourceView(buffer, first_element, num_elements, element_size).release()))
    {
    }

    GpuShaderResourceView::~GpuShaderResourceView() = default;

    GpuShaderResourceView::GpuShaderResourceView(GpuShaderResourceView&& other) noexcept = default;
    GpuShaderResourceView& GpuShaderResourceView::operator=(GpuShaderResourceView&& other) noexcept = default;

    void GpuShaderResourceView::Reset()
    {
        assert(impl_);
        impl_->Reset();
    }

    void GpuShaderResourceView::Transition(GpuCommandList& cmd_list) const
    {
        assert(impl_);
        impl_->Transition(cmd_list);
    }

    void GpuShaderResourceView::CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept
    {
        assert(impl_);
        impl_->CopyTo(dst_handle);
    }

    GpuDescriptorCpuHandle GpuShaderResourceView::CpuHandle() const noexcept
    {
        assert(impl_);
        return impl_->CpuHandle();
    }

    const GpuShaderResourceViewInternal& GpuShaderResourceView::Internal() const noexcept
    {
        assert(impl_);
        return *impl_;
    }


    class GpuRenderTargetView::Impl : public GpuRenderTargetViewInternal
    {
    };

    GpuRenderTargetView::GpuRenderTargetView() noexcept = default;

    GpuRenderTargetView::GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture)
        : GpuRenderTargetView(gpu_system, texture, GpuFormat::Unknown)
    {
    }
    GpuRenderTargetView::GpuRenderTargetView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateRenderTargetView(texture, format).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuRenderTargetViewInternal));
    }

    GpuRenderTargetView::~GpuRenderTargetView() = default;

    GpuRenderTargetView::GpuRenderTargetView(GpuRenderTargetView&& other) noexcept = default;
    GpuRenderTargetView& GpuRenderTargetView::operator=(GpuRenderTargetView&& other) noexcept = default;

    GpuRenderTargetView::operator bool() const noexcept
    {
        return impl_ && static_cast<bool>(*impl_);
    }

    void GpuRenderTargetView::Reset()
    {
        assert(impl_);
        impl_->Reset();
    }

    void GpuRenderTargetView::Transition(GpuCommandList& cmd_list) const
    {
        assert(impl_);
        impl_->Transition(cmd_list);
    }

    void GpuRenderTargetView::CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept
    {
        assert(impl_);
        impl_->CopyTo(dst_handle);
    }

    GpuDescriptorCpuHandle GpuRenderTargetView::CpuHandle() const noexcept
    {
        assert(impl_);
        return impl_->CpuHandle();
    }

    const GpuRenderTargetViewInternal& GpuRenderTargetView::Internal() const noexcept
    {
        assert(impl_);
        return *impl_;
    }


    class GpuDepthStencilView::Impl : public GpuDepthStencilViewInternal
    {
    };

    GpuDepthStencilView::GpuDepthStencilView() noexcept = default;

    GpuDepthStencilView::GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture)
        : GpuDepthStencilView(gpu_system, texture, GpuFormat::Unknown)
    {
    }
    GpuDepthStencilView::GpuDepthStencilView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateDepthStencilView(texture, format).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuDepthStencilViewInternal));
    }

    GpuDepthStencilView::~GpuDepthStencilView() = default;

    GpuDepthStencilView::GpuDepthStencilView(GpuDepthStencilView&& other) noexcept = default;
    GpuDepthStencilView& GpuDepthStencilView::operator=(GpuDepthStencilView&& other) noexcept = default;

    GpuDepthStencilView::operator bool() const noexcept
    {
        return impl_ && static_cast<bool>(*impl_);
    }

    void GpuDepthStencilView::Reset()
    {
        assert(impl_);
        impl_->Reset();
    }

    void GpuDepthStencilView::Transition(GpuCommandList& cmd_list) const
    {
        assert(impl_);
        impl_->Transition(cmd_list);
    }

    void GpuDepthStencilView::CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept
    {
        assert(impl_);
        impl_->CopyTo(dst_handle);
    }

    GpuDescriptorCpuHandle GpuDepthStencilView::CpuHandle() const noexcept
    {
        assert(impl_);
        return impl_->CpuHandle();
    }

    const GpuDepthStencilViewInternal& GpuDepthStencilView::Internal() const noexcept
    {
        assert(impl_);
        return *impl_;
    }


    class GpuUnorderedAccessView::Impl : public GpuUnorderedAccessViewInternal
    {
    };

    GpuUnorderedAccessView::GpuUnorderedAccessView() noexcept = default;

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture)
        : GpuUnorderedAccessView(gpu_system, texture, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, GpuFormat format)
        : GpuUnorderedAccessView(gpu_system, texture, 0, format)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource)
        : GpuUnorderedAccessView(gpu_system, texture, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateUnorderedAccessView(texture, sub_resource, format).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuUnorderedAccessViewInternal));
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array)
        : GpuUnorderedAccessView(gpu_system, texture_array, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, GpuFormat format)
        : GpuUnorderedAccessView(gpu_system, texture_array, 0, format)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource)
        : GpuUnorderedAccessView(gpu_system, texture_array, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(
        GpuSystem& gpu_system, GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateUnorderedAccessView(texture_array, sub_resource, format).release()))
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture)
        : GpuUnorderedAccessView(gpu_system, texture, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, GpuFormat format)
        : GpuUnorderedAccessView(gpu_system, texture, 0, format)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource)
        : GpuUnorderedAccessView(gpu_system, texture, sub_resource, GpuFormat::Unknown)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateUnorderedAccessView(texture, sub_resource, format).release()))
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, GpuFormat format)
        : GpuUnorderedAccessView(gpu_system, buffer, 0, buffer.Size() / FormatSize(format), format)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(
        GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateUnorderedAccessView(buffer, first_element, num_elements, format).release()))
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t element_size)
        : GpuUnorderedAccessView(gpu_system, buffer, 0, buffer.Size() / element_size, element_size)
    {
    }

    GpuUnorderedAccessView::GpuUnorderedAccessView(
        GpuSystem& gpu_system, GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size)
        : impl_(static_cast<Impl*>(
              gpu_system.Internal().CreateUnorderedAccessView(buffer, first_element, num_elements, element_size).release()))
    {
    }

    GpuUnorderedAccessView::~GpuUnorderedAccessView() = default;

    GpuUnorderedAccessView::GpuUnorderedAccessView(GpuUnorderedAccessView&& other) noexcept = default;
    GpuUnorderedAccessView& GpuUnorderedAccessView::operator=(GpuUnorderedAccessView&& other) noexcept = default;

    void GpuUnorderedAccessView::Reset()
    {
        assert(impl_);
        impl_->Reset();
    }

    void GpuUnorderedAccessView::Transition(GpuCommandList& cmd_list) const
    {
        assert(impl_);
        impl_->Transition(cmd_list);
    }

    void GpuUnorderedAccessView::CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept
    {
        assert(impl_);
        impl_->CopyTo(dst_handle);
    }

    GpuDescriptorCpuHandle GpuUnorderedAccessView::CpuHandle() const noexcept
    {
        assert(impl_);
        return impl_->CpuHandle();
    }

    GpuResource* GpuUnorderedAccessView::Resource() noexcept
    {
        return impl_ ? impl_->Resource() : nullptr;
    }

    const GpuResource* GpuUnorderedAccessView::Resource() const noexcept
    {
        return const_cast<GpuUnorderedAccessView*>(this)->Resource();
    }

    const GpuUnorderedAccessViewInternal& GpuUnorderedAccessView::Internal() const noexcept
    {
        assert(impl_);
        return *impl_;
    }
} // namespace AIHoloImager
