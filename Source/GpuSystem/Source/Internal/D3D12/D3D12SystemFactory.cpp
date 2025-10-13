// Copyright (c) 2025 Minmin Gong
//

#include "D3D12SystemFactory.hpp"

#include "D3D12/D3D12Conversion.hpp"

#include "D3D12Buffer.hpp"
#include "D3D12DescriptorHeap.hpp"
#include "D3D12Sampler.hpp"
#include "D3D12Texture.hpp"
#include "D3D12VertexAttrib.hpp"

namespace AIHoloImager
{
    D3D12SystemFactory::D3D12SystemFactory(GpuSystem& gpu_system) noexcept : gpu_system_(gpu_system)
    {
    }

    D3D12SystemFactory::~D3D12SystemFactory() = default;

    std::unique_ptr<GpuBufferInternal> D3D12SystemFactory::CreateBuffer(
        uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name) const
    {
        return std::make_unique<D3D12Buffer>(gpu_system_, size, heap, flags, std::move(name));
    }
    std::unique_ptr<GpuBufferInternal> D3D12SystemFactory::CreateBuffer(
        void* native_resource, GpuResourceState curr_state, std::wstring_view name) const
    {
        return std::make_unique<D3D12Buffer>(gpu_system_, native_resource, curr_state, std::move(name));
    }

    std::unique_ptr<GpuTextureInternal> D3D12SystemFactory::CreateTexture(GpuResourceType type, uint32_t width, uint32_t height,
        uint32_t depth, uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::wstring_view name) const
    {
        return std::make_unique<D3D12Texture>(
            gpu_system_, type, width, height, depth, array_size, mip_levels, format, flags, std::move(name));
    }
    std::unique_ptr<GpuTextureInternal> D3D12SystemFactory::CreateTexture(
        void* native_resource, GpuResourceState curr_state, std::wstring_view name) const
    {
        return std::make_unique<D3D12Texture>(gpu_system_, native_resource, curr_state, std::move(name));
    }

    std::unique_ptr<GpuStaticSamplerInternal> D3D12SystemFactory::CreateStaticSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<D3D12StaticSampler>(filters, addr_modes);
    }

    std::unique_ptr<GpuDynamicSamplerInternal> D3D12SystemFactory::CreateDynamicSampler(
        const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const
    {
        return std::make_unique<D3D12DynamicSampler>(gpu_system_, filters, addr_modes);
    }

    std::unique_ptr<GpuVertexAttribsInternal> D3D12SystemFactory::CreateVertexAttribs(std::span<const GpuVertexAttrib> attribs) const
    {
        return std::make_unique<D3D12VertexAttribs>(std::move(attribs));
    }

    std::unique_ptr<GpuDescriptorHeapInternal> D3D12SystemFactory::CreateDescriptorHeap(
        uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name) const
    {
        return std::make_unique<D3D12DescriptorHeap>(gpu_system_, size, type, shader_visible, std::move(name));
    }

    uint32_t D3D12SystemFactory::DescriptorSize(GpuDescriptorHeapType type) const
    {
        return gpu_system_.NativeDevice()->GetDescriptorHandleIncrementSize(ToD3D12DescriptorHeapType(type));
    }
} // namespace AIHoloImager
