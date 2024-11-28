// Copyright (c) 2024 Minmin Gong
//

#include "GpuSampler.hpp"

#include "Util/ErrorHandling.hpp"

namespace AIHoloImager
{
    GpuStaticSampler::Filters::Filters(Filter filter) : Filters(filter, filter, filter)
    {
    }

    GpuStaticSampler::Filters::Filters(Filter min_filter, Filter mag_filter) : Filters(min_filter, mag_filter, Filter::Point)
    {
    }

    GpuStaticSampler::Filters::Filters(Filter min_filter, Filter mag_filter, Filter mip_filter)
        : min(min_filter), mag(mag_filter), mip(mip_filter)
    {
    }


    GpuStaticSampler::AddressModes::AddressModes(AddressMode uvw) : AddressModes(uvw, uvw, uvw)
    {
    }

    GpuStaticSampler::AddressModes::AddressModes(AddressMode amu, AddressMode amv, AddressMode amw) : u(amu), v(amv), w(amw)
    {
    }


    GpuStaticSampler::GpuStaticSampler() noexcept
        : GpuStaticSampler({Filter::Point, Filter::Point, Filter::Point}, {AddressMode::Clamp, AddressMode::Clamp, AddressMode::Clamp})
    {
    }

    GpuStaticSampler::GpuStaticSampler(const Filters& filters, const AddressModes& addr_modes) : sampler_{}
    {
        auto to_d3d12_filter = [](const Filters& filters) {
            switch (filters.min)
            {
            case Filter::Point:
                switch (filters.mag)
                {
                case Filter::Point:
                    switch (filters.mip)
                    {
                    case Filter::Point:
                        return D3D12_FILTER_MIN_MAG_MIP_POINT;
                    case Filter::Linear:
                        return D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR;

                    default:
                        Unreachable("Invalid mip filter");
                    }
                    break;
                case Filter::Linear:
                    switch (filters.mip)
                    {
                    case Filter::Point:
                        return D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
                    case Filter::Linear:
                        return D3D12_FILTER_MIN_POINT_MAG_MIP_LINEAR;

                    default:
                        Unreachable("Invalid mip filter");
                    }
                    break;

                default:
                    Unreachable("Invalid mag filter");
                }
                break;
            case Filter::Linear:
                switch (filters.mag)
                {
                case Filter::Point:
                    switch (filters.mip)
                    {
                    case Filter::Point:
                        return D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT;
                    case Filter::Linear:
                        return D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR;

                    default:
                        Unreachable("Invalid mip filter");
                    }
                    break;
                case Filter::Linear:
                    switch (filters.mip)
                    {
                    case Filter::Point:
                        return D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                    case Filter::Linear:
                        return D3D12_FILTER_MIN_MAG_MIP_LINEAR;

                    default:
                        Unreachable("Invalid mip filter");
                    }
                    break;

                default:
                    Unreachable("Invalid mag filter");
                }
                break;

            default:
                Unreachable("Invalid min filter");
            }
        };
        sampler_.Filter = to_d3d12_filter(filters);

        auto to_d3d12_addr_mode = [](AddressMode mode) {
            switch (mode)
            {
            case AddressMode::Wrap:
                return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            case AddressMode::Mirror:
                return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            case AddressMode::Clamp:
                return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            case AddressMode::Border:
                return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            case AddressMode::MirrorOnce:
                return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;

            default:
                Unreachable("Invalid address mode");
            }
        };
        sampler_.AddressU = to_d3d12_addr_mode(addr_modes.u);
        sampler_.AddressV = to_d3d12_addr_mode(addr_modes.v);
        sampler_.AddressW = to_d3d12_addr_mode(addr_modes.w);

        sampler_.MaxAnisotropy = 16;
        sampler_.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        sampler_.MinLOD = 0.0f;
        sampler_.MaxLOD = D3D12_FLOAT32_MAX;
    }

    GpuStaticSampler::~GpuStaticSampler() = default;

    GpuStaticSampler::GpuStaticSampler(GpuStaticSampler&& other) noexcept = default;
    GpuStaticSampler& GpuStaticSampler::operator=(GpuStaticSampler&& other) noexcept = default;

    D3D12_STATIC_SAMPLER_DESC GpuStaticSampler::NativeStaticSampler(uint32_t register_index) const noexcept
    {
        D3D12_STATIC_SAMPLER_DESC ret = sampler_;
        ret.ShaderRegister = register_index;
        return ret;
    }
} // namespace AIHoloImager
