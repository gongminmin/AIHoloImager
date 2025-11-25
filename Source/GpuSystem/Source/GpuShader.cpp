// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuShader.hpp"

#include <cassert>

#include "Gpu/GpuCommandList.hpp"

#include "Internal/GpuShaderInternal.hpp"
#include "Internal/GpuSystemInternal.hpp"
#include "InternalImp.hpp"

namespace AIHoloImager
{
    EMPTY_IMP(GpuRenderPipeline)
    IMP_INTERNAL(GpuRenderPipeline)

    GpuRenderPipeline::GpuRenderPipeline() noexcept = default;
    GpuRenderPipeline::GpuRenderPipeline(GpuSystem& gpu_system, PrimitiveTopology topology, std::span<const ShaderInfo> shaders,
        const GpuVertexLayout& vertex_layout, std::span<const GpuStaticSampler> static_samplers, const States& states)
        : impl_(
              static_cast<Impl*>(gpu_system.Internal()
                                     .CreateRenderPipeline(topology, std::move(shaders), vertex_layout, std::move(static_samplers), states)
                                     .release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuRenderPipelineInternal));
    }

    GpuRenderPipeline::~GpuRenderPipeline() = default;

    GpuRenderPipeline::GpuRenderPipeline(GpuRenderPipeline&& other) noexcept = default;
    GpuRenderPipeline& GpuRenderPipeline::operator=(GpuRenderPipeline&& other) noexcept = default;

    void GpuRenderPipeline::Bind(GpuCommandList& cmd_list) const
    {
        assert(impl_);
        impl_->Bind(cmd_list);
    }


    EMPTY_IMP(GpuComputePipeline)
    IMP_INTERNAL(GpuComputePipeline)

    GpuComputePipeline::GpuComputePipeline() noexcept = default;
    GpuComputePipeline::GpuComputePipeline(
        GpuSystem& gpu_system, const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers)
        : impl_(static_cast<Impl*>(gpu_system.Internal().CreateComputePipeline(shader, std::move(static_samplers)).release()))
    {
        static_assert(sizeof(Impl) == sizeof(GpuComputePipelineInternal));
    }

    GpuComputePipeline::~GpuComputePipeline() = default;

    GpuComputePipeline::GpuComputePipeline(GpuComputePipeline&& other) noexcept = default;
    GpuComputePipeline& GpuComputePipeline::operator=(GpuComputePipeline&& other) noexcept = default;

    void GpuComputePipeline::Bind(GpuCommandList& cmd_list) const
    {
        assert(impl_);
        impl_->Bind(cmd_list);
    }
} // namespace AIHoloImager