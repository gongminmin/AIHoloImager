// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>
#include <span>

#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuVertexAttrib.hpp"

#include "GpuResourceInternal.hpp"
#include "GpuSamplerInternal.hpp"
#include "GpuVertexAttribInternal.hpp"

namespace AIHoloImager
{
    class GpuSystemInternalFactory
    {
    public:
        virtual ~GpuSystemInternalFactory();

        virtual std::unique_ptr<GpuResourceInternal> CreateGpuResource() const = 0;
        virtual std::unique_ptr<GpuResourceInternal> CreateGpuResource(void* native_resource, std::wstring_view name) const = 0;

        virtual std::unique_ptr<GpuStaticSamplerInternal> CreateGpuStaticSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const = 0;
        virtual std::unique_ptr<GpuDynamicSamplerInternal> CreateGpuDynamicSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const = 0;

        virtual std::unique_ptr<GpuVertexAttribsInternal> CreateGpuVertexAttribs(std::span<const GpuVertexAttrib> attribs) const = 0;
    };
} // namespace AIHoloImager
