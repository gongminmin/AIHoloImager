// Copyright (c) 2025 Minmin Gong
//

#include "GpuResourceViewsInternal.hpp"

namespace AIHoloImager
{
    GpuConstantBufferViewInternal::GpuConstantBufferViewInternal() noexcept = default;
    GpuConstantBufferViewInternal::~GpuConstantBufferViewInternal() = default;

    GpuConstantBufferViewInternal::GpuConstantBufferViewInternal(GpuConstantBufferViewInternal&& other) noexcept = default;
    GpuConstantBufferViewInternal& GpuConstantBufferViewInternal::operator=(GpuConstantBufferViewInternal&& other) noexcept = default;


    GpuShaderResourceViewInternal::GpuShaderResourceViewInternal() noexcept = default;
    GpuShaderResourceViewInternal::~GpuShaderResourceViewInternal() = default;

    GpuShaderResourceViewInternal::GpuShaderResourceViewInternal(GpuShaderResourceViewInternal&& other) noexcept = default;
    GpuShaderResourceViewInternal& GpuShaderResourceViewInternal::operator=(GpuShaderResourceViewInternal&& other) noexcept = default;


    GpuRenderTargetViewInternal::GpuRenderTargetViewInternal() noexcept = default;
    GpuRenderTargetViewInternal::~GpuRenderTargetViewInternal() = default;

    GpuRenderTargetViewInternal::GpuRenderTargetViewInternal(GpuRenderTargetViewInternal&& other) noexcept = default;
    GpuRenderTargetViewInternal& GpuRenderTargetViewInternal::operator=(GpuRenderTargetViewInternal&& other) noexcept = default;


    GpuDepthStencilViewInternal::GpuDepthStencilViewInternal() noexcept = default;
    GpuDepthStencilViewInternal::~GpuDepthStencilViewInternal() = default;

    GpuDepthStencilViewInternal::GpuDepthStencilViewInternal(GpuDepthStencilViewInternal&& other) noexcept = default;
    GpuDepthStencilViewInternal& GpuDepthStencilViewInternal::operator=(GpuDepthStencilViewInternal&& other) noexcept = default;


    GpuUnorderedAccessViewInternal::GpuUnorderedAccessViewInternal() noexcept = default;
    GpuUnorderedAccessViewInternal::~GpuUnorderedAccessViewInternal() = default;

    GpuUnorderedAccessViewInternal::GpuUnorderedAccessViewInternal(GpuUnorderedAccessViewInternal&& other) noexcept = default;
    GpuUnorderedAccessViewInternal& GpuUnorderedAccessViewInternal::operator=(GpuUnorderedAccessViewInternal&& other) noexcept = default;
} // namespace AIHoloImager
