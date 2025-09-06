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

    struct GpuSampler
    {
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

            Filters(Filter filter);
            Filters(Filter min_filter, Filter mag_filter);
            Filters(Filter min_filter, Filter mag_filter, Filter mip_filter);
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

            AddressModes(AddressMode uvw);
            AddressModes(AddressMode amu, AddressMode amv, AddressMode amw);
        };
    };

    class GpuStaticSampler : public GpuSampler
    {
        DISALLOW_COPY_AND_ASSIGN(GpuStaticSampler)

    public:
        GpuStaticSampler() noexcept;
        GpuStaticSampler(const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes);
        ~GpuStaticSampler();

        GpuStaticSampler(GpuStaticSampler&& other) noexcept;
        GpuStaticSampler& operator=(GpuStaticSampler&& other) noexcept;

        D3D12_STATIC_SAMPLER_DESC NativeStaticSampler(uint32_t register_index) const noexcept;

    private:
        D3D12_STATIC_SAMPLER_DESC sampler_;
    };

    class GpuDynamicSampler : public GpuSampler
    {
        DISALLOW_COPY_AND_ASSIGN(GpuDynamicSampler)

    public:
        explicit GpuDynamicSampler(GpuSystem& gpu_system) noexcept;
        GpuDynamicSampler(GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes);
        ~GpuDynamicSampler();

        GpuDynamicSampler(GpuDynamicSampler&& other) noexcept;
        GpuDynamicSampler& operator=(GpuDynamicSampler&& other) noexcept;

        void CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept;

    private:
        GpuSystem* gpu_system_ = nullptr;

        GpuDescriptorBlock desc_block_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_{};
    };
} // namespace AIHoloImager
