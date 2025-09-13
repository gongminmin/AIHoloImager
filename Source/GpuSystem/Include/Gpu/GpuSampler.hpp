// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <cstdint>

#include <directx/d3d12.h>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuDescriptorAllocator.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class GpuSampler
    {
    public:
        enum class Filter
        {
            Point,
            Linear,
        };

        struct Filters
        {
            Filter min;
            Filter mag;
            Filter mip;

            Filters() noexcept;
            Filters(Filter filter) noexcept;
            Filters(Filter min_filter, Filter mag_filter) noexcept;
            Filters(Filter min_filter, Filter mag_filter, Filter mip_filter) noexcept;
        };

        enum class AddressMode
        {
            Wrap,
            Mirror,
            Clamp,
            Border,
            MirrorOnce,
        };

        struct AddressModes
        {
            AddressMode u;
            AddressMode v;
            AddressMode w;

            AddressModes() noexcept;
            AddressModes(AddressMode uvw) noexcept;
            AddressModes(AddressMode amu, AddressMode amv, AddressMode amw) noexcept;
        };

        const GpuSampler::Filters& SamplerFilters() const noexcept
        {
            return filters_;
        }
        const GpuSampler::AddressModes& SamplerAddressModes() const noexcept
        {
            return addr_modes_;
        }

    protected:
        GpuSampler() noexcept;
        GpuSampler(const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) noexcept;
        ~GpuSampler() noexcept;

        GpuSampler::Filters filters_;
        GpuSampler::AddressModes addr_modes_;
    };

    class GpuStaticSampler final : public GpuSampler
    {
        DISALLOW_COPY_AND_ASSIGN(GpuStaticSampler)

    public:
        GpuStaticSampler() noexcept;
        GpuStaticSampler(const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes);
        ~GpuStaticSampler() noexcept;

        GpuStaticSampler(GpuStaticSampler&& other) noexcept;
        GpuStaticSampler& operator=(GpuStaticSampler&& other) noexcept;

        D3D12_STATIC_SAMPLER_DESC NativeStaticSampler(uint32_t register_index) const noexcept;

    private:
        D3D12_STATIC_SAMPLER_DESC sampler_{};
    };

    class GpuDynamicSampler final : public GpuSampler
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDynamicSampler)

    public:
        GpuDynamicSampler() noexcept;
        explicit GpuDynamicSampler(GpuSystem& gpu_system);
        GpuDynamicSampler(GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes);
        ~GpuDynamicSampler() noexcept;

        GpuDynamicSampler(GpuDynamicSampler&& other) noexcept;
        GpuDynamicSampler& operator=(GpuDynamicSampler&& other) noexcept;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;

        const D3D12_SAMPLER_DESC& NativeSampler() const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;

        GpuDescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};

        D3D12_SAMPLER_DESC sampler_{};
    };
} // namespace AIHoloImager
