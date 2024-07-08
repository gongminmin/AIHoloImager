// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>
#include <span>

#include <directx/d3d12.h>

#include "GpuSystem.hpp"
#include "Util/ComPtr.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuRenderPipeline
    {
        DISALLOW_COPY_AND_ASSIGN(GpuRenderPipeline)

    public:
        enum class ShaderStage
        {
            Vertex = 0,
            Pixel,

            Num,
        };

        struct ShaderInfo
        {
            ShaderStage stage;
            std::span<const uint8_t> bytecode;
            uint32_t num_cbs;
            uint32_t num_srvs;
            uint32_t num_uavs;
        };

        enum CullMode
        {
            None,
            ClockWise,
            CounterClockWise,
        };

        // TODO: Support more states
        struct States
        {
            CullMode cull_mode = CullMode::ClockWise;
            bool conservative_raster = false;
            bool depth_enable = false;
            std::span<const DXGI_FORMAT> rtv_formats;
            DXGI_FORMAT dsv_format;
        };

    public:
        GpuRenderPipeline() noexcept;
        GpuRenderPipeline(GpuSystem& gpu_system, std::span<const ShaderInfo> shaders, std::span<const D3D12_STATIC_SAMPLER_DESC> samplers,
            const States& states, std::span<const D3D12_INPUT_ELEMENT_DESC> input_elems);
        ~GpuRenderPipeline() noexcept;

        GpuRenderPipeline(GpuRenderPipeline&& other) noexcept;
        GpuRenderPipeline& operator=(GpuRenderPipeline&& other) noexcept;

        ID3D12RootSignature* NativeRootSignature() const noexcept;
        ID3D12PipelineState* NativePipelineState() const noexcept;

    private:
        ComPtr<ID3D12RootSignature> root_sig_;
        ComPtr<ID3D12PipelineState> pso_;
    };

    class GpuComputeShader
    {
        DISALLOW_COPY_AND_ASSIGN(GpuComputeShader)

    public:
        GpuComputeShader() noexcept;
        GpuComputeShader(GpuSystem& gpu_system, std::span<const uint8_t> bytecode, uint32_t num_cbs, uint32_t num_srvs, uint32_t num_uavs,
            std::span<const D3D12_STATIC_SAMPLER_DESC> samplers);
        ~GpuComputeShader() noexcept;

        GpuComputeShader(GpuComputeShader&& other) noexcept;
        GpuComputeShader& operator=(GpuComputeShader&& other) noexcept;

        ID3D12RootSignature* NativeRootSignature() const noexcept;
        ID3D12PipelineState* NativePipelineState() const noexcept;

    private:
        ComPtr<ID3D12RootSignature> root_sig_;
        ComPtr<ID3D12PipelineState> pso_;
    };
} // namespace AIHoloImager
