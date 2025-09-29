// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuSampler.hpp"

#include "Base/ErrorHandling.hpp"
#include "Gpu/GpuSystem.hpp"

#include "D3D12/D3D12Conversion.hpp"

namespace AIHoloImager
{
    template <typename T>
    void FillSamplerDesc(T& sampler, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes)
    {
        sampler = {};

        auto to_d3d12_filter = [](const GpuSampler::Filters& filters) {
            switch (filters.min)
            {
            case GpuSampler::Filter::Point:
                switch (filters.mag)
                {
                case GpuSampler::Filter::Point:
                    switch (filters.mip)
                    {
                    case GpuSampler::Filter::Point:
                        return D3D12_FILTER_MIN_MAG_MIP_POINT;
                    case GpuSampler::Filter::Linear:
                        return D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR;

                    default:
                        Unreachable("Invalid mip filter");
                    }
                    break;
                case GpuSampler::Filter::Linear:
                    switch (filters.mip)
                    {
                    case GpuSampler::Filter::Point:
                        return D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
                    case GpuSampler::Filter::Linear:
                        return D3D12_FILTER_MIN_POINT_MAG_MIP_LINEAR;

                    default:
                        Unreachable("Invalid mip filter");
                    }
                    break;

                default:
                    Unreachable("Invalid mag filter");
                }
                break;
            case GpuSampler::Filter::Linear:
                switch (filters.mag)
                {
                case GpuSampler::Filter::Point:
                    switch (filters.mip)
                    {
                    case GpuSampler::Filter::Point:
                        return D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT;
                    case GpuSampler::Filter::Linear:
                        return D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR;

                    default:
                        Unreachable("Invalid mip filter");
                    }
                    break;
                case GpuSampler::Filter::Linear:
                    switch (filters.mip)
                    {
                    case GpuSampler::Filter::Point:
                        return D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                    case GpuSampler::Filter::Linear:
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
        sampler.Filter = to_d3d12_filter(filters);

        auto to_d3d12_addr_mode = [](GpuSampler::AddressMode mode) {
            switch (mode)
            {
            case GpuSampler::AddressMode::Wrap:
                return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            case GpuSampler::AddressMode::Mirror:
                return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            case GpuSampler::AddressMode::Clamp:
                return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            case GpuSampler::AddressMode::Border:
                return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            case GpuSampler::AddressMode::MirrorOnce:
                return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;

            default:
                Unreachable("Invalid address mode");
            }
        };
        sampler.AddressU = to_d3d12_addr_mode(addr_modes.u);
        sampler.AddressV = to_d3d12_addr_mode(addr_modes.v);
        sampler.AddressW = to_d3d12_addr_mode(addr_modes.w);

        sampler.MaxAnisotropy = 16;
        sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler.MinLOD = 0.0f;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
    }


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


    GpuStaticSampler::GpuStaticSampler() noexcept
        : GpuStaticSampler({Filter::Point, Filter::Point, Filter::Point}, {AddressMode::Clamp, AddressMode::Clamp, AddressMode::Clamp})
    {
    }
    GpuStaticSampler::GpuStaticSampler(const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes)
        : GpuSampler(filters, addr_modes)
    {
        FillSamplerDesc(sampler_, filters, addr_modes);
    }

    GpuStaticSampler::~GpuStaticSampler() noexcept = default;

    GpuStaticSampler::GpuStaticSampler(GpuStaticSampler&& other) noexcept = default;
    GpuStaticSampler& GpuStaticSampler::operator=(GpuStaticSampler&& other) noexcept = default;

    D3D12_STATIC_SAMPLER_DESC GpuStaticSampler::NativeStaticSampler(uint32_t register_index) const noexcept
    {
        D3D12_STATIC_SAMPLER_DESC ret = sampler_;
        ret.ShaderRegister = register_index;
        return ret;
    }


    GpuDynamicSampler::GpuDynamicSampler() noexcept = default;
    GpuDynamicSampler::GpuDynamicSampler(GpuSystem& gpu_system)
        : GpuDynamicSampler(
              gpu_system, {Filter::Point, Filter::Point, Filter::Point}, {AddressMode::Clamp, AddressMode::Clamp, AddressMode::Clamp})
    {
    }
    GpuDynamicSampler::GpuDynamicSampler(GpuSystem& gpu_system, const Filters& filters, const AddressModes& addr_modes)
        : GpuSampler(filters, addr_modes), gpu_system_(&gpu_system)
    {
        desc_block_ = gpu_system.AllocSamplerDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        FillSamplerDesc(sampler_, filters, addr_modes);
        gpu_system_->NativeDevice()->CreateSampler(&sampler_, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    GpuDynamicSampler::~GpuDynamicSampler() noexcept = default;

    GpuDynamicSampler::GpuDynamicSampler(GpuDynamicSampler&& other) noexcept = default;
    GpuDynamicSampler& GpuDynamicSampler::operator=(GpuDynamicSampler&& other) noexcept = default;

    void GpuDynamicSampler::CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept
    {
        gpu_system_->NativeDevice()->CopyDescriptorsSimple(
            1, ToD3D12CpuDescriptorHandle(dst_handle), ToD3D12CpuDescriptorHandle(cpu_handle_), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    }

    const D3D12_SAMPLER_DESC& GpuDynamicSampler::NativeSampler() const noexcept
    {
        return sampler_;
    }
} // namespace AIHoloImager
