// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>
#include <span>

#include <directx/d3d12.h>

#include "../GpuSystemInternalFactory.hpp"
#include "Gpu/GpuVertexAttrib.hpp"

namespace AIHoloImager
{
    class GpuSystem;

    class D3D12SystemFactory : public GpuSystemInternalFactory
    {
    public:
        explicit D3D12SystemFactory(GpuSystem& gpu_system) noexcept;
        ~D3D12SystemFactory() override;

        std::unique_ptr<GpuBufferInternal> CreateBuffer(
            uint32_t size, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name) const override;
        std::unique_ptr<GpuBufferInternal> CreateBuffer(
            void* native_resource, GpuResourceState curr_state, std::wstring_view name) const override;

        std::unique_ptr<GpuTextureInternal> CreateTexture(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth,
            uint32_t array_size, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::wstring_view name) const override;
        std::unique_ptr<GpuTextureInternal> CreateTexture(
            void* native_resource, GpuResourceState curr_state, std::wstring_view name) const override;

        std::unique_ptr<GpuStaticSamplerInternal> CreateStaticSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const override;
        std::unique_ptr<GpuDynamicSamplerInternal> CreateDynamicSampler(
            const GpuSampler::Filters& filters, const GpuSampler::AddressModes& addr_modes) const override;

        std::unique_ptr<GpuVertexAttribsInternal> CreateVertexAttribs(std::span<const GpuVertexAttrib> attribs) const override;

        std::unique_ptr<GpuDescriptorHeapInternal> CreateDescriptorHeap(
            uint32_t size, GpuDescriptorHeapType type, bool shader_visible, std::wstring_view name) const override;

        uint32_t DescriptorSize(GpuDescriptorHeapType type) const override;

    private:
        GpuSystem& gpu_system_;
    };
} // namespace AIHoloImager
