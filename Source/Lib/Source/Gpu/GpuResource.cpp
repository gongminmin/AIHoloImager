// Copyright (c) 2024 Minmin Gong
//

#include "GpuResource.hpp"

#include "Util/ErrorHandling.hpp"

namespace AIHoloImager
{
    D3D12_HEAP_TYPE ToD3D12HeapType(GpuHeap heap)
    {
        switch (heap)
        {
        case GpuHeap::Default:
            return D3D12_HEAP_TYPE_DEFAULT;
        case GpuHeap::Upload:
            return D3D12_HEAP_TYPE_UPLOAD;
        case GpuHeap::ReadBack:
            return D3D12_HEAP_TYPE_READBACK;

        default:
            Unreachable("Invalid heap");
        }
    }

    D3D12_RESOURCE_FLAGS ToD3D12ResourceFlags(GpuResourceFlag flags)
    {
        D3D12_RESOURCE_FLAGS d3d12_flag = D3D12_RESOURCE_FLAG_NONE;
        if (EnumHasAny(flags, GpuResourceFlag::RenderTarget))
        {
            d3d12_flag |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }
        if (EnumHasAny(flags, GpuResourceFlag::DepthStencil))
        {
            d3d12_flag |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        }
        if (EnumHasAny(flags, GpuResourceFlag::UnorderedAccess))
        {
            d3d12_flag |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        return d3d12_flag;
    }
} // namespace AIHoloImager
