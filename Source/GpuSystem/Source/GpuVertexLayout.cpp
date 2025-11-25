// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuVertexLayout.hpp"

#include "Gpu/GpuSystem.hpp"

#include "Internal/GpuSystemInternal.hpp"
#include "Internal/GpuVertexLayoutInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    EMPTY_IMP(GpuVertexLayout)
    IMP_INTERNAL(GpuVertexLayout)

    GpuVertexLayout::GpuVertexLayout() noexcept = default;

    GpuVertexLayout::GpuVertexLayout(
        GpuSystem& gpu_system, std::span<const GpuVertexAttrib> attribs, std::span<const uint32_t> slot_strides)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateVertexLayout(std::move(attribs), std::move(slot_strides)).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuVertexLayoutInternal));
    }

    GpuVertexLayout::GpuVertexLayout(const GpuVertexLayout& other) : impl_(static_cast<Impl*>(other.impl_->Clone().release()))
    {
    }

    GpuVertexLayout::~GpuVertexLayout() noexcept = default;

    GpuVertexLayout& GpuVertexLayout::operator=(const GpuVertexLayout& other)
    {
        if (this != &other)
        {
            if (impl_)
            {
                static_cast<GpuVertexLayoutInternal&>(*impl_) = static_cast<GpuVertexLayoutInternal&>(*other.impl_);
            }
            else
            {
                impl_.reset(static_cast<Impl*>(other.impl_->Clone().release()));
            }
        }
        return *this;
    }

    GpuVertexLayout::GpuVertexLayout(GpuVertexLayout&& other) noexcept : impl_(std::move(other.impl_))
    {
    }

    GpuVertexLayout& GpuVertexLayout::operator=(GpuVertexLayout&& other) noexcept
    {
        if (this != &other)
        {
            impl_ = std::move(other.impl_);
        }
        return *this;
    }
} // namespace AIHoloImager
