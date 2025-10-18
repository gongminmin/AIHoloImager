// Copyright (c) 2025 Minmin Gong
//

#include "D3D12SystemFactory.hpp"

#include "D3D12/D3D12Conversion.hpp"
#include "Gpu/D3D12/D3D12Traits.hpp"

#include "D3D12Buffer.hpp"
#include "D3D12CommandAllocatorInfo.hpp"
#include "D3D12CommandList.hpp"
#include "D3D12DescriptorHeap.hpp"
#include "D3D12ResourceViews.hpp"
#include "D3D12Sampler.hpp"
#include "D3D12Shader.hpp"
#include "D3D12System.hpp"
#include "D3D12Texture.hpp"
#include "D3D12VertexAttrib.hpp"

namespace AIHoloImager
{
    D3D12SystemFactory::D3D12SystemFactory(GpuSystem& gpu_system) noexcept : gpu_system_(gpu_system)
    {
    }

    D3D12SystemFactory::~D3D12SystemFactory() noexcept = default;

    std::unique_ptr<GpuSystemInternal> D3D12SystemFactory::CreateSystem(
        std::function<bool(void* device)> confirm_device, bool enable_sharing, bool enable_debug) const
    {
        return std::make_unique<D3D12System>(gpu_system_, std::move(confirm_device), enable_sharing, enable_debug);
    }

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
        return gpu_system_.NativeDevice<D3D12Traits>()->GetDescriptorHandleIncrementSize(ToD3D12DescriptorHeapType(type));
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12SystemFactory::CreateShaderResourceView(
        const GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12SystemFactory::CreateShaderResourceView(
        const GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(gpu_system_, texture_array, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12SystemFactory::CreateShaderResourceView(
        const GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12SystemFactory::CreateShaderResourceView(
        const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const
    {
        return std::make_unique<D3D12ShaderResourceView>(gpu_system_, buffer, first_element, num_elements, format);
    }

    std::unique_ptr<GpuShaderResourceViewInternal> D3D12SystemFactory::CreateShaderResourceView(
        const GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const
    {
        return std::make_unique<D3D12ShaderResourceView>(gpu_system_, buffer, first_element, num_elements, element_size);
    }

    std::unique_ptr<GpuRenderTargetViewInternal> D3D12SystemFactory::CreateRenderTargetView(GpuTexture2D& texture, GpuFormat format) const
    {
        return std::make_unique<D3D12RenderTargetView>(gpu_system_, texture, format);
    }

    std::unique_ptr<GpuDepthStencilViewInternal> D3D12SystemFactory::CreateDepthStencilView(GpuTexture2D& texture, GpuFormat format) const
    {
        return std::make_unique<D3D12DepthStencilView>(gpu_system_, texture, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12SystemFactory::CreateUnorderedAccessView(
        GpuTexture2D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12SystemFactory::CreateUnorderedAccessView(
        GpuTexture2DArray& texture_array, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(gpu_system_, texture_array, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12SystemFactory::CreateUnorderedAccessView(
        GpuTexture3D& texture, uint32_t sub_resource, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(gpu_system_, texture, sub_resource, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12SystemFactory::CreateUnorderedAccessView(
        GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, GpuFormat format) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(gpu_system_, buffer, first_element, num_elements, format);
    }

    std::unique_ptr<GpuUnorderedAccessViewInternal> D3D12SystemFactory::CreateUnorderedAccessView(
        GpuBuffer& buffer, uint32_t first_element, uint32_t num_elements, uint32_t element_size) const
    {
        return std::make_unique<D3D12UnorderedAccessView>(gpu_system_, buffer, first_element, num_elements, element_size);
    }

    std::unique_ptr<GpuRenderPipelineInternal> D3D12SystemFactory::CreateRenderPipeline(GpuRenderPipeline::PrimitiveTopology topology,
        std::span<const ShaderInfo> shaders, const GpuVertexAttribs& vertex_attribs, std::span<const GpuStaticSampler> static_samplers,
        const GpuRenderPipeline::States& states) const
    {
        return std::make_unique<D3D12RenderPipeline>(
            gpu_system_, topology, std::move(shaders), vertex_attribs, std::move(static_samplers), states);
    }

    std::unique_ptr<GpuComputePipelineInternal> D3D12SystemFactory::CreateComputePipeline(
        const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers) const
    {
        return std::make_unique<D3D12ComputePipeline>(gpu_system_, shader, std::move(static_samplers));
    }

    std::unique_ptr<GpuCommandAllocatorInfoInternal> D3D12SystemFactory::CreateCommandAllocatorInfo(GpuSystem::CmdQueueType type) const
    {
        return std::make_unique<D3D12CommandAllocatorInfo>(gpu_system_, type);
    }

    std::unique_ptr<GpuCommandListInternal> D3D12SystemFactory::CreateCommandList(
        GpuCommandAllocatorInfo& cmd_alloc_info, GpuSystem::CmdQueueType type) const
    {
        return std::make_unique<D3D12CommandList>(gpu_system_, cmd_alloc_info, type);
    }
} // namespace AIHoloImager
