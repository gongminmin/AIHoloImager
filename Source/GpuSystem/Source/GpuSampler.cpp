// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuSampler.hpp"

#include "Gpu/GpuSystem.hpp"

#include "Internal/GpuSamplerInternal.hpp"
#include "Internal/GpuSystemInternalFactory.hpp"

namespace AIHoloImager
{
    GpuSampler::Filters::Filters() noexcept : Filters(Filter::Point)
    {
    }
    GpuSampler::Filters::Filters(Filter filter) noexcept : Filters(filter, filter, filter)
    {
    }
    GpuSampler::Filters::Filters(Filter min_filter, Filter mag_filter) noexcept : Filters(min_filter, mag_filter, Filter::Point)
    {
    }
    GpuSampler::Filters::Filters(Filter min_filter, Filter mag_filter, Filter mip_filter) noexcept
        : min(min_filter), mag(mag_filter), mip(mip_filter)
    {
    }


    GpuSampler::AddressModes::AddressModes() noexcept : AddressModes(AddressMode::Wrap)
    {
    }
    GpuSampler::AddressModes::AddressModes(AddressMode uvw) noexcept : AddressModes(uvw, uvw, uvw)
    {
    }
    GpuSampler::AddressModes::AddressModes(AddressMode amu, AddressMode amv, AddressMode amw) noexcept : u(amu), v(amv), w(amw)
    {
    }


    GpuSampler::GpuSampler() noexcept = default;
    GpuSampler::GpuSampler(const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) noexcept
        : filters_(filters), addr_modes_(addr_modes)
    {
    }
    GpuSampler::~GpuSampler() noexcept = default;


    class GpuStaticSampler::Impl : public GpuStaticSamplerInternal
    {
    };

    GpuStaticSampler::GpuStaticSampler() noexcept = default;
    GpuStaticSampler::GpuStaticSampler(GpuSystem& gpu_system)
        : GpuStaticSampler(
              gpu_system, {Filter::Point, Filter::Point, Filter::Point}, {AddressMode::Clamp, AddressMode::Clamp, AddressMode::Clamp})
    {
    }
    GpuStaticSampler::GpuStaticSampler(
        GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes)
        : GpuSampler(filters, addr_modes),
          impl_(static_cast<Impl*>(gpu_system.InternalFactory().CreateStaticSampler(filters, addr_modes).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuStaticSamplerInternal));
    }

    GpuStaticSampler::~GpuStaticSampler() noexcept = default;

    GpuStaticSampler::GpuStaticSampler(GpuStaticSampler&& other) noexcept = default;
    GpuStaticSampler& GpuStaticSampler::operator=(GpuStaticSampler&& other) noexcept = default;

    const GpuStaticSamplerInternal& GpuStaticSampler::Internal() const noexcept
    {
        return *impl_;
    }


    class GpuDynamicSampler::Impl : public GpuDynamicSamplerInternal
    {
    };

    GpuDynamicSampler::GpuDynamicSampler() noexcept = default;
    GpuDynamicSampler::GpuDynamicSampler(GpuSystem& gpu_system)
        : GpuDynamicSampler(
              gpu_system, {Filter::Point, Filter::Point, Filter::Point}, {AddressMode::Clamp, AddressMode::Clamp, AddressMode::Clamp})
    {
    }
    GpuDynamicSampler::GpuDynamicSampler(GpuSystem& gpu_system, const Filters& filters, const AddressModes& addr_modes)
        : GpuSampler(filters, addr_modes),
          impl_(static_cast<Impl*>(gpu_system.InternalFactory().CreateDynamicSampler(filters, addr_modes).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuDynamicSamplerInternal));
    }

    GpuDynamicSampler::~GpuDynamicSampler() noexcept = default;

    GpuDynamicSampler::GpuDynamicSampler(GpuDynamicSampler&& other) noexcept = default;
    GpuDynamicSampler& GpuDynamicSampler::operator=(GpuDynamicSampler&& other) noexcept = default;

    const GpuDynamicSamplerInternal& GpuDynamicSampler::Internal() const noexcept
    {
        return *impl_;
    }
} // namespace AIHoloImager
