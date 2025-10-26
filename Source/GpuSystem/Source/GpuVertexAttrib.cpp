// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuVertexAttrib.hpp"

#include "Gpu/GpuSystem.hpp"

#include "Internal/GpuSystemInternal.hpp"
#include "Internal/GpuVertexAttribInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    EMPTY_IMP(GpuVertexAttribs)
    IMP_INTERNAL(GpuVertexAttribs)

    GpuVertexAttribs::GpuVertexAttribs() noexcept = default;

    GpuVertexAttribs::GpuVertexAttribs(GpuSystem& gpu_system, std::span<const GpuVertexAttrib> attribs)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateVertexAttribs(std::move(attribs)).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuVertexAttribsInternal));
    }

    GpuVertexAttribs::GpuVertexAttribs(const GpuVertexAttribs& other) : impl_(static_cast<Impl*>(other.impl_->Clone().release()))
    {
    }

    GpuVertexAttribs::~GpuVertexAttribs() noexcept = default;

    GpuVertexAttribs& GpuVertexAttribs::operator=(const GpuVertexAttribs& other)
    {
        if (this != &other)
        {
            if (impl_)
            {
                static_cast<GpuVertexAttribsInternal&>(*impl_) = static_cast<GpuVertexAttribsInternal&>(*other.impl_);
            }
            else
            {
                impl_.reset(static_cast<Impl*>(other.impl_->Clone().release()));
            }
        }
        return *this;
    }

    GpuVertexAttribs::GpuVertexAttribs(GpuVertexAttribs&& other) noexcept : impl_(std::move(other.impl_))
    {
    }

    GpuVertexAttribs& GpuVertexAttribs::operator=(GpuVertexAttribs&& other) noexcept
    {
        if (this != &other)
        {
            impl_ = std::move(other.impl_);
        }
        return *this;
    }
} // namespace AIHoloImager
