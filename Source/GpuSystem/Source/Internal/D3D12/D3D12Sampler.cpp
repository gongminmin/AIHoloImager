// Copyright (c) 2025 Minmin Gong
//

#include "D3D12Sampler.hpp"

#include "Base/ErrorHandling.hpp"
#include "Gpu/D3D12/D3D12Traits.hpp"
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

    D3D12StaticSampler::D3D12StaticSampler(const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes)
    {
        FillSamplerDesc(sampler_, filters, addr_modes);
    }

    D3D12StaticSampler::~D3D12StaticSampler() noexcept = default;

    D3D12StaticSampler::D3D12StaticSampler(D3D12StaticSampler&& other) noexcept = default;
    D3D12StaticSampler::D3D12StaticSampler(GpuStaticSamplerInternal&& other) noexcept
        : D3D12StaticSampler(static_cast<D3D12StaticSampler&&>(other))
    {
    }

    D3D12StaticSampler& D3D12StaticSampler::operator=(D3D12StaticSampler&& other) noexcept = default;
    GpuStaticSamplerInternal& D3D12StaticSampler::operator=(GpuStaticSamplerInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12StaticSampler&&>(other));
    }

    D3D12_STATIC_SAMPLER_DESC D3D12StaticSampler::SamplerDesc(uint32_t register_index) const noexcept
    {
        D3D12_STATIC_SAMPLER_DESC ret = sampler_;
        ret.ShaderRegister = register_index;
        return ret;
    }


    D3D12DynamicSampler::D3D12DynamicSampler(
        GpuSystem& gpu_system, const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes)
        : gpu_system_(&gpu_system)
    {
        desc_block_ = gpu_system.AllocSamplerDescBlock(1);
        cpu_handle_ = desc_block_.CpuHandle();

        FillSamplerDesc(sampler_, filters, addr_modes);
        gpu_system_->NativeDevice<D3D12Traits>()->CreateSampler(&sampler_, ToD3D12CpuDescriptorHandle(cpu_handle_));
    }

    D3D12DynamicSampler::~D3D12DynamicSampler() noexcept = default;

    D3D12DynamicSampler::D3D12DynamicSampler(D3D12DynamicSampler&& other) noexcept = default;
    D3D12DynamicSampler::D3D12DynamicSampler(GpuDynamicSamplerInternal&& other) noexcept
        : D3D12DynamicSampler(static_cast<D3D12DynamicSampler&&>(other))
    {
    }

    D3D12DynamicSampler& D3D12DynamicSampler::operator=(D3D12DynamicSampler&& other) noexcept = default;
    GpuDynamicSamplerInternal& D3D12DynamicSampler::operator=(GpuDynamicSamplerInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12DynamicSampler&&>(other));
    }

    void D3D12DynamicSampler::CopyTo(GpuDescriptorCpuHandle dst_handle) const noexcept
    {
        gpu_system_->NativeDevice<D3D12Traits>()->CopyDescriptorsSimple(
            1, ToD3D12CpuDescriptorHandle(dst_handle), ToD3D12CpuDescriptorHandle(cpu_handle_), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    }

    const D3D12_SAMPLER_DESC& D3D12DynamicSampler::SamplerDesc() const noexcept
    {
        return sampler_;
    }
} // namespace AIHoloImager
