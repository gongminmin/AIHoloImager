// Copyright (c) 2025 Minmin Gong
//

#pragma once

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

    D3D12_CPU_DESCRIPTOR_HANDLE ToD3D12CpuDescriptorHandle(GpuDescriptorCpuHandle handle) noexcept;
    GpuDescriptorCpuHandle FromD3D12CpuDescriptorHandle(D3D12_CPU_DESCRIPTOR_HANDLE handle) noexcept;

    D3D12_GPU_DESCRIPTOR_HANDLE ToD3D12GpuDescriptorHandle(GpuDescriptorGpuHandle handle) noexcept;
    GpuDescriptorGpuHandle FromD3D12GpuDescriptorHandle(D3D12_GPU_DESCRIPTOR_HANDLE handle) noexcept;

    D3D12_DESCRIPTOR_HEAP_TYPE ToD3D12DescriptorHeapType(GpuDescriptorHeapType type) noexcept;
    GpuDescriptorHeapType FromD3D12DescriptorHeapType(D3D12_DESCRIPTOR_HEAP_TYPE type) noexcept;

    D3D_PRIMITIVE_TOPOLOGY ToD3D12PrimitiveTopology(GpuRenderPipeline::PrimitiveTopology topology) noexcept;
} // namespace AIHoloImager
