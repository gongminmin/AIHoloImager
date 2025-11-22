// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Gpu/GpuDescriptorAllocator.hpp"
#include "Gpu/GpuSampler.hpp"

#include "../GpuSamplerInternal.hpp"
#include "D3D12ImpDefine.hpp"

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

    D3D12_DEFINE_IMP(StaticSampler)

    class D3D12DynamicSampler : public GpuDynamicSamplerInternal
    {
    public:
        D3D12DynamicSampler(GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes);
        ~D3D12DynamicSampler() override;

        D3D12DynamicSampler(D3D12DynamicSampler&& other) noexcept;
        explicit D3D12DynamicSampler(GpuDynamicSamplerInternal&& other) noexcept;

        D3D12DynamicSampler& operator=(D3D12DynamicSampler&& other) noexcept;
        GpuDynamicSamplerInternal& operator=(GpuDynamicSamplerInternal&& other) noexcept override;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;

        const D3D12_SAMPLER_DESC& SamplerDesc() const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;
        ID3D12Device* d3d12_device_;

        GpuDescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};

        D3D12_SAMPLER_DESC sampler_{};
    };

    D3D12_DEFINE_IMP(DynamicSampler)
} // namespace AIHoloImager
