// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuShader.hpp"

#include <cassert>

#include "Gpu/GpuCommandList.hpp"

#include "Internal/GpuShaderInternal.hpp"
#include "Internal/GpuSystemInternalFactory.hpp"

namespace AIHoloImager
{
    class GpuRenderPipeline::Impl : public GpuRenderPipelineInternal
    {
    };

    GpuRenderPipeline::GpuRenderPipeline() noexcept = default;
    GpuRenderPipeline::GpuRenderPipeline(GpuSystem& gpu_system, PrimitiveTopology topology, std::span<const ShaderInfo> shaders,
        const GpuVertexAttribs& vertex_attribs, std::span<const GpuStaticSampler> static_samplers, const States& states)
        : impl_(
              static_cast<Impl*>(gpu_system.InternalFactory()
                                     .CreateRenderPipeline(topology, std::move(shaders), vertex_attribs, std::move(static_samplers), states)
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

    const GpuRenderPipelineInternal& GpuRenderPipeline::Internal() const noexcept
    {
        assert(impl_);
        return *impl_;
    }


    class GpuComputePipeline::Impl : public GpuComputePipelineInternal
    {
    };

    GpuComputePipeline::GpuComputePipeline() noexcept = default;
    GpuComputePipeline::GpuComputePipeline(
        GpuSystem& gpu_system, const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers)
        : impl_(static_cast<Impl*>(gpu_system.InternalFactory().CreateComputePipeline(shader, std::move(static_samplers)).release()))
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

    const GpuComputePipelineInternal& GpuComputePipeline::Internal() const noexcept
    {
        assert(impl_);
        return *impl_;
    }
} // namespace AIHoloImager