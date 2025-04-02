// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <cstdint>
#include <span>

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Base/Noncopyable.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuUtil.hpp"
#include "Gpu/GpuVertexAttrib.hpp"

namespace AIHoloImager
{
    struct ShaderInfo
    {
        std::span<const uint8_t> bytecode;
        uint32_t num_cbs = 0;
        uint32_t num_srvs = 0;
        uint32_t num_uavs = 0;
    };

    class GpuRenderPipeline
    {
        DISALLOW_COPY_AND_ASSIGN(GpuRenderPipeline)

    public:
        enum class ShaderStage
        {
            Vertex = 0,
            Pixel,
            Geometry,

            Num,
        };

        enum class CullMode
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
            std::span<const GpuFormat> rtv_formats;
            GpuFormat dsv_format = GpuFormat::Unknown;
        };

        // TODO: Support more topology
        enum class PrimitiveTopology
        {
            PointList,
            TriangleList,
            TriangleStrip,
        };

    public:
        GpuRenderPipeline() noexcept;
        GpuRenderPipeline(GpuSystem& gpu_system, PrimitiveTopology topology, std::span<const ShaderInfo> shaders,
            const GpuVertexAttribs& vertex_attribs, std::span<const GpuStaticSampler> samplers, const States& states);
        ~GpuRenderPipeline();

        GpuRenderPipeline(GpuRenderPipeline&& other) noexcept;
        GpuRenderPipeline& operator=(GpuRenderPipeline&& other) noexcept;

        ID3D12RootSignature* NativeRootSignature() const noexcept;
        ID3D12PipelineState* NativePipelineState() const noexcept;
        D3D_PRIMITIVE_TOPOLOGY NativePrimitiveTopology() const noexcept;

    private:
        GpuRecyclableObject<ComPtr<ID3D12RootSignature>> root_sig_;
        GpuRecyclableObject<ComPtr<ID3D12PipelineState>> pso_;
        D3D_PRIMITIVE_TOPOLOGY topology_;
    };

    class GpuComputePipeline
    {
        DISALLOW_COPY_AND_ASSIGN(GpuComputePipeline)

    public:
        GpuComputePipeline() noexcept;
        GpuComputePipeline(GpuSystem& gpu_system, const ShaderInfo& shader, std::span<const GpuStaticSampler> samplers);
        ~GpuComputePipeline();

        GpuComputePipeline(GpuComputePipeline&& other) noexcept;
        GpuComputePipeline& operator=(GpuComputePipeline&& other) noexcept;

        ID3D12RootSignature* NativeRootSignature() const noexcept;
        ID3D12PipelineState* NativePipelineState() const noexcept;

    private:
        GpuRecyclableObject<ComPtr<ID3D12RootSignature>> root_sig_;
        GpuRecyclableObject<ComPtr<ID3D12PipelineState>> pso_;
    };
} // namespace AIHoloImager
