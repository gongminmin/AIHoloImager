// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResource.hpp"

#include "GpuResourceInternal.hpp"

namespace AIHoloImager
{
    class GpuTextureInternal : public GpuResourceInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTextureInternal)

    public:
        GpuTextureInternal() noexcept;
        virtual ~GpuTextureInternal();

        GpuTextureInternal(GpuTextureInternal&& other) noexcept;
        virtual GpuTextureInternal& operator=(GpuTextureInternal&& other) noexcept = 0;

        virtual void* NativeTexture() const noexcept = 0;

        virtual uint32_t Width(uint32_t mip) const noexcept = 0;
        virtual uint32_t Height(uint32_t mip) const noexcept = 0;
        virtual uint32_t Depth(uint32_t mip) const noexcept = 0;
        virtual uint32_t ArraySize() const noexcept = 0;
        virtual uint32_t MipLevels() const noexcept = 0;
        virtual uint32_t Planes() const noexcept = 0;
        virtual GpuFormat Format() const noexcept = 0;
        virtual GpuResourceFlag Flags() const noexcept = 0;
    };
} // namespace AIHoloImager
