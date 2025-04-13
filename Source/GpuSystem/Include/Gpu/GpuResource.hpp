// Copyright (c) 2024-2005 Minmin Gong
//

#pragma once

#include <cstdint>

#include <directx/d3d12.h>

#include "Base/Enum.hpp"

namespace AIHoloImager
{
    enum class GpuHeap
    {
        Default,
        Upload,
        ReadBack,
    };

    D3D12_HEAP_TYPE ToD3D12HeapType(GpuHeap heap);

    enum class GpuResourceFlag : uint32_t
    {
        None = 0,
        RenderTarget = 1U << 0,
        DepthStencil = 1U << 1,
        UnorderedAccess = 1U << 2,
    };
    ENUM_CLASS_BITWISE_OPERATORS(GpuResourceFlag);

    D3D12_RESOURCE_FLAGS ToD3D12ResourceFlags(GpuResourceFlag flags) noexcept;

    enum class GpuResourceState
    {
        Common,

        ColorWrite,
        DepthWrite,

        UnorderedAccess,

        CopySrc,
        CopyDst,

        RayTracingAS,
    };
    D3D12_RESOURCE_STATES ToD3D12ResourceState(GpuResourceState state);
} // namespace AIHoloImager
