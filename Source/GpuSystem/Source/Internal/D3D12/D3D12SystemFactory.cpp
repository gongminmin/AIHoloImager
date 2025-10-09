// Copyright (c) 2025 Minmin Gong
//

#include "D3D12SystemFactory.hpp"

#include "D3D12Resource.hpp"
#include "D3D12Sampler.hpp"
#include "D3D12VertexAttrib.hpp"

namespace AIHoloImager
{
    D3D12SystemFactory::D3D12SystemFactory(GpuSystem& gpu_system) noexcept : gpu_system_(gpu_system)
    {
    }

    D3D12SystemFactory::~D3D12SystemFactory() = default;

    std::unique_ptr<GpuResourceInternal> D3D12SystemFactory::CreateGpuResource() const
    {
        return std::make_unique<D3D12Resource>(gpu_system_);
    }
    std::unique_ptr<GpuResourceInternal> D3D12SystemFactory::CreateGpuResource(void* native_resource, std::wstring_view name) const
    {
        return std::make_unique<D3D12Resource>(gpu_system_, native_resource, name);
    }

    std::unique_ptr<GpuStaticSamplerInternal> D3D12SystemFactory::CreateGpuStaticSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<D3D12StaticSampler>(filters, addr_modes);
    }

    std::unique_ptr<GpuDynamicSamplerInternal> D3D12SystemFactory::CreateGpuDynamicSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<D3D12DynamicSampler>(gpu_system_, filters, addr_modes);
    }

    std::unique_ptr<GpuVertexAttribsInternal> D3D12SystemFactory::CreateGpuVertexAttribs(std::span<const GpuVertexAttrib> attribs) const
    {
        return std::make_unique<D3D12VertexAttribs>(std::move(attribs));
    }
} // namespace AIHoloImager
