// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuCommandList.hpp"

namespace AIHoloImager
{
    class GpuRenderPipelineInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuRenderPipelineInternal)

    public:
        GpuRenderPipelineInternal() noexcept;
        virtual ~GpuRenderPipelineInternal();

        GpuRenderPipelineInternal(GpuRenderPipelineInternal&& other) noexcept;
        virtual GpuRenderPipelineInternal& operator=(GpuRenderPipelineInternal&& other) noexcept = 0;

        virtual void Bind(GpuCommandList& cmd_list) const = 0;
    };

    class GpuComputePipelineInternal
    {
        DISALLOW_COPY_AND_ASSIGN(GpuComputePipelineInternal)

    public:
        GpuComputePipelineInternal() noexcept;
        virtual ~GpuComputePipelineInternal();

        GpuComputePipelineInternal(GpuComputePipelineInternal&& other) noexcept;
        virtual GpuComputePipelineInternal& operator=(GpuComputePipelineInternal&& other) noexcept = 0;

        virtual void Bind(GpuCommandList& cmd_list) const = 0;
    };
} // namespace AIHoloImager
