// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <cstdint>

#include <directx/d3d12.h>

#include "Base/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuStaticSampler
    {
        DISALLOW_COPY_AND_ASSIGN(GpuStaticSampler)

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

    public:
        GpuStaticSampler() noexcept;
        GpuStaticSampler(const Filters& filters, const AddressModes& addr_modes);
        ~GpuStaticSampler();

        GpuStaticSampler(GpuStaticSampler&& other) noexcept;
        GpuStaticSampler& operator=(GpuStaticSampler&& other) noexcept;

        D3D12_STATIC_SAMPLER_DESC NativeStaticSampler(uint32_t register_index) const noexcept;

    private:
        D3D12_STATIC_SAMPLER_DESC sampler_;
    };
} // namespace AIHoloImager
