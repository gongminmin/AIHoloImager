// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuVertexLayout.hpp"
#include "Gpu/InternalDefine.hpp"

#ifdef AIHI_ENABLE_D3D12
    #ifdef AIHI_ENABLE_VULKAN
        #define DEFINE_SHADER(name) \
            {                       \
                name##_dxil,        \
                name##_spv,         \
            },                      \
                #name,
    #else
        #define DEFINE_SHADER(name) \
            {                       \
                name##_dxil,        \
                {},                 \
            },                      \
                #name,
    #endif
#else
    #ifdef AIHI_ENABLE_VULKAN
        #define DEFINE_SHADER(name) \
            {                       \
                {},                 \
                name##_spv,         \
            },                      \
                #name,
    #endif
#endif

namespace AIHoloImager
{
    struct ShaderInfo
    {
        enum class BytecodeFormat
        {
            Dxil = 0,
            Spv,

            Num,
        };

        std::span<const uint8_t> bytecodes[static_cast<uint32_t>(BytecodeFormat::Num)];
        std::string_view name;
    };

    class GpuRenderPipelineInternal;

    class GpuRenderPipeline
    {
        DISALLOW_COPY_AND_ASSIGN(GpuRenderPipeline)
        DEFINE_INTERNAL(GpuRenderPipeline)

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
            const GpuVertexLayout& vertex_layout, std::span<const GpuStaticSampler> static_samplers, const States& states);
        ~GpuRenderPipeline();

        GpuRenderPipeline(GpuRenderPipeline&& other) noexcept;
        GpuRenderPipeline& operator=(GpuRenderPipeline&& other) noexcept;

        void Bind(GpuCommandList& cmd_list) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class GpuComputePipelineInternal;

    class GpuComputePipeline
    {
        DISALLOW_COPY_AND_ASSIGN(GpuComputePipeline)
        DEFINE_INTERNAL(GpuComputePipeline)

    public:
        GpuComputePipeline() noexcept;
        GpuComputePipeline(GpuSystem& gpu_system, const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers);
        ~GpuComputePipeline();

        GpuComputePipeline(GpuComputePipeline&& other) noexcept;
        GpuComputePipeline& operator=(GpuComputePipeline&& other) noexcept;

        void Bind(GpuCommandList& cmd_list) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
