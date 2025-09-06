// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuSampler.hpp"

#include "Base/ErrorHandling.hpp"
#include "Gpu/GpuSystem.hpp"

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


    GpuSampler::Filters::Filters(Filter filter) : Filters(filter, filter, filter)
    {
    }

    GpuSampler::Filters::Filters(Filter min_filter, Filter mag_filter) : Filters(min_filter, mag_filter, Filter::Point)
    {
    }

    GpuSampler::Filters::Filters(Filter min_filter, Filter mag_filter, Filter mip_filter)
        : min(min_filter), mag(mag_filter), mip(mip_filter)
    {
    }


    GpuSampler::AddressModes::AddressModes(AddressMode uvw) : AddressModes(uvw, uvw, uvw)
    {
    }

    GpuSampler::AddressModes::AddressModes(AddressMode amu, AddressMode amv, AddressMode amw) : u(amu), v(amv), w(amw)
    {
    }


    GpuStaticSampler::GpuStaticSampler() noexcept
        : GpuStaticSampler({Filter::Point, Filter::Point, Filter::Point}, {AddressMode::Clamp, AddressMode::Clamp, AddressMode::Clamp})
    {
    }

    GpuStaticSampler::GpuStaticSampler(const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) : sampler_{}
    {
        FillSamplerDesc(sampler_, filters, addr_modes);
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


    GpuDynamicSampler::GpuDynamicSampler(GpuSystem& gpu_system) noexcept
        : GpuDynamicSampler(
              gpu_system, {Filter::Point, Filter::Point, Filter::Point}, {AddressMode::Clamp, AddressMode::Clamp, AddressMode::Clamp})
    {
    }

    GpuDynamicSampler::GpuDynamicSampler(GpuSystem& gpu_system, const Filters& filters, const AddressModes& addr_modes)
        : gpu_system_(&gpu_system)
    {
        desc_block_ = gpu_system.AllocSamplerDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        D3D12_SAMPLER_DESC sampler;
        FillSamplerDesc(sampler, filters, addr_modes);
        gpu_system_->NativeDevice()->CreateSampler(&sampler, cpu_handle_);
    }

    GpuDynamicSampler::~GpuDynamicSampler() = default;

    GpuDynamicSampler::GpuDynamicSampler(GpuDynamicSampler&& other) noexcept = default;
    GpuDynamicSampler& GpuDynamicSampler::operator=(GpuDynamicSampler&& other) noexcept = default;

    void GpuDynamicSampler::CopyTo(D3D12_CPU_DESCRIPTOR_HANDLE dst_handle) const noexcept
    {
        gpu_system_->NativeDevice()->CopyDescriptorsSimple(1, dst_handle, cpu_handle_, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    }
} // namespace AIHoloImager
