// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>
#include <span>

#include "../GpuSystemInternalFactory.hpp"
#include "Gpu/GpuVertexAttrib.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class D3D12SystemFactory : public GpuSystemInternalFactory
    {
    public:
        explicit D3D12SystemFactory(GpuSystem& gpu_system) noexcept;
        ~D3D12SystemFactory() override;

        std::unique_ptr<GpuResourceInternal> CreateGpuResource() const override;
        std::unique_ptr<GpuResourceInternal> CreateGpuResource(void* native_resource, std::wstring_view name) const override;

        std::unique_ptr<GpuStaticSamplerInternal> CreateGpuStaticSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const override;
        std::unique_ptr<GpuDynamicSamplerInternal> CreateGpuDynamicSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const override;

        std::unique_ptr<GpuVertexAttribsInternal> CreateGpuVertexAttribs(std::span<const GpuVertexAttrib> attribs) const override;

    private:
        GpuSystem& gpu_system_;
    };
} // namespace AIHoloImager
