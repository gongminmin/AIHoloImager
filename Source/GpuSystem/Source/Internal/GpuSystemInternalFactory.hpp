// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>
#include <span>

#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuVertexAttrib.hpp"

#include "GpuBufferInternal.hpp"
#include "GpuSamplerInternal.hpp"
#include "GpuTextureInternal.hpp"
#include "GpuVertexAttribInternal.hpp"

namespace AIHoloImager
{
    class GpuSystemInternalFactory
    {
    public:
        virtual ~GpuSystemInternalFactory();

        virtual std::unique_ptr<GpuBufferInternal> CreateBuffer(
            uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name) const = 0;
        virtual std::unique_ptr<GpuBufferInternal> CreateBuffer(
            void* native_resource, GpuResourceState curr_state, std::wstring_view name) const = 0;

        virtual std::unique_ptr<GpuTextureInternal> CreateTexture(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
            uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::wstring_view name) const = 0;
        virtual std::unique_ptr<GpuTextureInternal> CreateTexture(
            void* native_resource, GpuResourceState curr_state, std::wstring_view name) const = 0;

        virtual std::unique_ptr<GpuStaticSamplerInternal> CreateStaticSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const = 0;
        virtual std::unique_ptr<GpuDynamicSamplerInternal> CreateDynamicSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const = 0;

        virtual std::unique_ptr<GpuVertexAttribsInternal> CreateVertexAttribs(std::span<const GpuVertexAttrib> attribs) const = 0;
    };
} // namespace AIHoloImager
