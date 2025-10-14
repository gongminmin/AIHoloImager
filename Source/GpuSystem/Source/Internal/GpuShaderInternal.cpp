// Copyright (c) 2025 Minmin Gong
//

#include "GpuShaderInternal.hpp"

namespace AIHoloImager
{
    GpuRenderPipelineInternal::GpuRenderPipelineInternal() noexcept = default;
    GpuRenderPipelineInternal::~GpuRenderPipelineInternal() = default;

    GpuRenderPipelineInternal::GpuRenderPipelineInternal(GpuRenderPipelineInternal&& other) noexcept = default;
    GpuRenderPipelineInternal& GpuRenderPipelineInternal::operator=(GpuRenderPipelineInternal&& other) noexcept = default;


    GpuComputePipelineInternal::GpuComputePipelineInternal() noexcept = default;
    GpuComputePipelineInternal::~GpuComputePipelineInternal() = default;

    GpuComputePipelineInternal::GpuComputePipelineInternal(GpuComputePipelineInternal&& other) noexcept = default;
    GpuComputePipelineInternal& GpuComputePipelineInternal::operator=(GpuComputePipelineInternal&& other) noexcept = default;
} // namespace AIHoloImager
