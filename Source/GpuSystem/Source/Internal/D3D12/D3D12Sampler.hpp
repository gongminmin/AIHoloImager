// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <directx/d3d12.h>

#include "../GpuSamplerInternal.hpp"
#include "Gpu/GpuDescriptorAllocator.hpp"
#include "Gpu/GpuSampler.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class D3D12StaticSampler : public GpuStaticSamplerInternal
    {
    public:
        D3D12StaticSampler(const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes);
        ~D3D12StaticSampler() override;

        D3D12StaticSampler(D3D12StaticSampler&& other) noexcept;
        explicit D3D12StaticSampler(GpuStaticSamplerInternal&& other) noexcept;

        D3D12StaticSampler& operator=(D3D12StaticSampler&& other) noexcept;
        GpuStaticSamplerInternal& operator=(GpuStaticSamplerInternal&& other) noexcept override;

        D3D12_STATIC_SAMPLER_DESC SamplerDesc(uint32_t register_index) const noexcept;

    private:
        D3D12_STATIC_SAMPLER_DESC sampler_{};
    };

    class D3D12DynamicSampler : public GpuDynamicSamplerInternal
    {
    public:
        D3D12DynamicSampler(GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes);
        ~D3D12DynamicSampler() override;

        D3D12DynamicSampler(D3D12DynamicSampler&& other) noexcept;
        explicit D3D12DynamicSampler(GpuDynamicSamplerInternal&& other) noexcept;

        D3D12DynamicSampler& operator=(D3D12DynamicSampler&& other) noexcept;
        GpuDynamicSamplerInternal& operator=(GpuDynamicSamplerInternal&& other) noexcept override;

        void CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept;

        const D3D12_SAMPLER_DESC& SamplerDesc() const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;

        GpuDescriptorBlock desc_block_;
        GpuDescriptorCpuHandle cpu_handle_{};

        D3D12_SAMPLER_DESC sampler_{};
    };
} // namespace AIHoloImager
