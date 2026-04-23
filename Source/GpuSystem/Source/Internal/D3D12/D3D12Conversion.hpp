// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>
#include <directx/dxgiformat.h>

#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResource.hpp"
#include "Gpu/GpuShader.hpp"

namespace AIHoloImager
{
    DXGI_FORMAT ToDxgiFormat(GpuFormat fmt);
    GpuFormat FromDxgiFormat(DXGI_FORMAT fmt);

    D3D12_HEAP_TYPE ToD3D12HeapType(GpuHeap heap);
    GpuHeap FromD3D12HeapType(D3D12_HEAP_TYPE heap_type);

    D3D12_RESOURCE_FLAGS ToD3D12ResourceFlags(GpuResourceFlag flags) noexcept;
    GpuResourceFlag FromD3D12ResourceFlags(D3D12_RESOURCE_FLAGS flags) noexcept;

    D3D12_HEAP_FLAGS ToD3D12HeapFlags(GpuResourceFlag flags) noexcept;

    D3D12_RESOURCE_STATES ToD3D12ResourceState(GpuResourceState state);

    D3D12_RESOURCE_DIMENSION ToD3D12ResourceDimension(GpuResourceType type);

    D3D_PRIMITIVE_TOPOLOGY ToD3D12PrimitiveTopology(GpuRenderPipeline::PrimitiveTopology topology) noexcept;

    D3D12_BLEND ToD3D12Blend(GpuRenderPipeline::BlendFactor blend) noexcept;
    D3D12_BLEND_OP ToD3D12BlendOp(GpuRenderPipeline::BlendOp blend_op) noexcept;
} // namespace AIHoloImager
