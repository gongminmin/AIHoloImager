// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/Noncopyable.hpp"

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

    class GpuStaticSamplerInternal;

    class GpuStaticSampler final : public GpuSampler
    {
        DISALLOW_COPY_AND_ASSIGN(GpuStaticSampler)

    public:
        GpuStaticSampler() noexcept;
        explicit GpuStaticSampler(GpuSystem& gpu_system);
        GpuStaticSampler(GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes);
        ~GpuStaticSampler() noexcept;

        GpuStaticSampler(GpuStaticSampler&& other) noexcept;
        GpuStaticSampler& operator=(GpuStaticSampler&& other) noexcept;

        const GpuStaticSamplerInternal& Internal() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class GpuDynamicSamplerInternal;

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

        const GpuDynamicSamplerInternal& Internal() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
