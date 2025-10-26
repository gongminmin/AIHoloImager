// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <span>

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuShader.hpp"

#include "../GpuShaderInternal.hpp"
#include "D3D12CommandList.hpp"
#include "D3D12ImpDefine.hpp"
#include "D3D12Util.hpp"

namespace AIHoloImager
{
    class D3D12RenderPipeline : public GpuRenderPipelineInternal
    {
    public:
        D3D12RenderPipeline(GpuSystem& gpu_system, GpuRenderPipeline::PrimitiveTopology topology, std::span<const ShaderInfo> shaders,
            const GpuVertexAttribs& vertex_attribs, std::span<const GpuStaticSampler> static_samplers,
            const GpuRenderPipeline::States& states);
        ~D3D12RenderPipeline() override;

        D3D12RenderPipeline(D3D12RenderPipeline&& other) noexcept;
        explicit D3D12RenderPipeline(GpuRenderPipelineInternal&& other) noexcept;
        D3D12RenderPipeline& operator=(D3D12RenderPipeline&& other) noexcept;
        GpuRenderPipelineInternal& operator=(GpuRenderPipelineInternal&& other) noexcept override;

        void Bind(GpuCommandList& cmd_list) const override;
        void Bind(D3D12CommandList& cmd_list) const;

    private:
        D3D12RecyclableObject<ComPtr<ID3D12RootSignature>> root_sig_;
        D3D12RecyclableObject<ComPtr<ID3D12PipelineState>> pso_;
        GpuRenderPipeline::PrimitiveTopology topology_{};
    };

    D3D12_DEFINE_IMP(RenderPipeline)

    class D3D12ComputePipeline : public GpuComputePipelineInternal
    {
    public:
        D3D12ComputePipeline(GpuSystem& gpu_system, const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers);
        ~D3D12ComputePipeline() override;

        D3D12ComputePipeline(D3D12ComputePipeline&& other) noexcept;
        explicit D3D12ComputePipeline(GpuComputePipelineInternal&& other) noexcept;
        D3D12ComputePipeline& operator=(D3D12ComputePipeline&& other) noexcept;
        GpuComputePipelineInternal& operator=(GpuComputePipelineInternal&& other) noexcept override;

        void Bind(GpuCommandList& cmd_list) const override;
        void Bind(D3D12CommandList& cmd_list) const;

    private:
        D3D12RecyclableObject<ComPtr<ID3D12RootSignature>> root_sig_;
        D3D12RecyclableObject<ComPtr<ID3D12PipelineState>> pso_;
    };

    D3D12_DEFINE_IMP(ComputePipeline)
} // namespace AIHoloImager
