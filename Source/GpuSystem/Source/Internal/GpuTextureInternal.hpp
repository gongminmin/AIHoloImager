// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResource.hpp"

namespace AIHoloImager
{
    class GpuTextureInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTextureInternal)

    public:
        GpuTextureInternal() noexcept;
        virtual ~GpuTextureInternal();

        GpuTextureInternal(GpuTextureInternal&& other) noexcept;
        virtual GpuTextureInternal& operator=(GpuTextureInternal&& other) noexcept = 0;

        virtual void Name(std::wstring_view name) = 0;

        virtual void* NativeResource() const noexcept = 0;
        virtual void* NativeTexture() const noexcept = 0;

        virtual void* SharedHandle() const noexcept = 0;

        virtual uint32_t Width(uint32_t mip) const noexcept = 0;
        virtual uint32_t Height(uint32_t mip) const noexcept = 0;
        virtual uint32_t Depth(uint32_t mip) const noexcept = 0;
        virtual uint32_t ArraySize() const noexcept = 0;
        virtual uint32_t MipLevels() const noexcept = 0;
        virtual uint32_t Planes() const noexcept = 0;
        virtual GpuFormat Format() const noexcept = 0;
        virtual GpuResourceFlag Flags() const noexcept = 0;

        virtual void Reset() = 0;

        virtual void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const = 0;
        virtual void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const = 0;
    };
} // namespace AIHoloImager
