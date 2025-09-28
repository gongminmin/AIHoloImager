// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <directx/d3d12.h>
#include <directx/dxgiformat.h>

#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResource.hpp"

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
} // namespace AIHoloImager
