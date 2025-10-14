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

        std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const override;
        std::unique_ptr<GpuShaderResourceViewInternal> CreateShaderResourceView(
            const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const override;

        std::unique_ptr<GpuRenderTargetViewInternal> CreateRenderTargetView(GpuTexture2D& texture, GpuFormat format) const override;

        std::unique_ptr<GpuDepthStencilViewInternal> CreateDepthStencilView(GpuTexture2D& texture, GpuFormat format) const override;

        std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const override;
        std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const override;
        std::unique_ptr<GpuUnorderedAccessViewInternal> CreateUnorderedAccessView(
            GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const override;

        std::unique_ptr<GpuRenderPipelineInternal> CreateRenderPipeline(GpuRenderPipeline::PrimitiveTopology topology,
            std::span<const ShaderInfo> shaders, const GpuVertexAttribs& vertex_attribs, std::span<const GpuStaticSampler> static_samplers,
            const GpuRenderPipeline::States& states) const override;
        std::unique_ptr<GpuComputePipelineInternal> CreateComputePipeline(
            const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers) const override;

    private:
        GpuSystem& gpu_system_;
    };
} // namespace AIHoloImager
