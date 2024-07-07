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
