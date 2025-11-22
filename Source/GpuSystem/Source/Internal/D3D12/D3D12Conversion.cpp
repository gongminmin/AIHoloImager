// Copyright (c) 2025 Minmin Gong
//

#include "D3D12Conversion.hpp"

#include "Base/ErrorHandling.hpp"

namespace AIHoloImager
{
    DXGI_FORMAT ToDxgiFormat(GpuFormat fmt)
    {
        switch (fmt)
        {
        case GpuFormat::Unknown:
            return DXGI_FORMAT_UNKNOWN;

        case GpuFormat::R8_UNorm:
            return DXGI_FORMAT_R8_UNORM;

        case GpuFormat::RG8_UNorm:
            return DXGI_FORMAT_R8G8_UNORM;

        case GpuFormat::RGBA8_UNorm:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case GpuFormat::RGBA8_UNorm_SRGB:
            return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case GpuFormat::BGRA8_UNorm:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case GpuFormat::BGRA8_UNorm_SRGB:
            return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        case GpuFormat::BGRX8_UNorm:
            return DXGI_FORMAT_B8G8R8X8_UNORM;
        case GpuFormat::BGRX8_UNorm_SRGB:
            return DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;

        case GpuFormat::R16_Uint:
            return DXGI_FORMAT_R16_UINT;
        case GpuFormat::R16_Sint:
            return DXGI_FORMAT_R16_SINT;
        case GpuFormat::R16_Float:
            return DXGI_FORMAT_R16_FLOAT;

        case GpuFormat::RG16_Uint:
            return DXGI_FORMAT_R16G16_UINT;
        case GpuFormat::RG16_Sint:
            return DXGI_FORMAT_R16G16_SINT;
        case GpuFormat::RG16_Float:
            return DXGI_FORMAT_R16G16_FLOAT;

        case GpuFormat::RGBA16_Uint:
            return DXGI_FORMAT_R16G16B16A16_UINT;
        case GpuFormat::RGBA16_Sint:
            return DXGI_FORMAT_R16G16B16A16_SINT;
        case GpuFormat::RGBA16_Float:
            return DXGI_FORMAT_R16G16B16A16_FLOAT;

        case GpuFormat::R32_Uint:
            return DXGI_FORMAT_R32_UINT;
        case GpuFormat::R32_Sint:
            return DXGI_FORMAT_R32_SINT;
        case GpuFormat::R32_Float:
            return DXGI_FORMAT_R32_FLOAT;

        case GpuFormat::RG32_Uint:
            return DXGI_FORMAT_R32G32_UINT;
        case GpuFormat::RG32_Sint:
            return DXGI_FORMAT_R32G32_SINT;
        case GpuFormat::RG32_Float:
            return DXGI_FORMAT_R32G32_FLOAT;

        case GpuFormat::RGB32_Uint:
            return DXGI_FORMAT_R32G32B32_UINT;
        case GpuFormat::RGB32_Sint:
            return DXGI_FORMAT_R32G32B32_SINT;
        case GpuFormat::RGB32_Float:
            return DXGI_FORMAT_R32G32B32_FLOAT;

        case GpuFormat::RGBA32_Uint:
            return DXGI_FORMAT_R32G32B32A32_UINT;
        case GpuFormat::RGBA32_Sint:
            return DXGI_FORMAT_R32G32B32A32_SINT;
        case GpuFormat::RGBA32_Float:
            return DXGI_FORMAT_R32G32B32A32_FLOAT;

        case GpuFormat::D16_UNorm:
            return DXGI_FORMAT_D16_UNORM;
        case GpuFormat::D24_UNorm_S8_Uint:
            return DXGI_FORMAT_D24_UNORM_S8_UINT;
        case GpuFormat::D32_Float:
            return DXGI_FORMAT_D32_FLOAT;
        case GpuFormat::D32_Float_S8X24_Uint:
            return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;

        default:
            Unreachable("Invalid format");
        }
    }

    GpuFormat FromDxgiFormat(DXGI_FORMAT fmt)
    {
        switch (fmt)
        {
        case DXGI_FORMAT_UNKNOWN:
            return GpuFormat::Unknown;

        case DXGI_FORMAT_R8_UNORM:
            return GpuFormat::R8_UNorm;

        case DXGI_FORMAT_R8G8_UNORM:
            return GpuFormat::RG8_UNorm;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
            return GpuFormat::RGBA8_UNorm;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return GpuFormat::RGBA8_UNorm_SRGB;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
            return GpuFormat::BGRA8_UNorm;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return GpuFormat::BGRA8_UNorm_SRGB;
        case DXGI_FORMAT_B8G8R8X8_UNORM:
            return GpuFormat::BGRX8_UNorm;
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
            return GpuFormat::BGRX8_UNorm_SRGB;

        case DXGI_FORMAT_R16_UINT:
            return GpuFormat::R16_Uint;
        case DXGI_FORMAT_R16_SINT:
            return GpuFormat::R16_Sint;
        case DXGI_FORMAT_R16_FLOAT:
            return GpuFormat::R16_Float;

        case DXGI_FORMAT_R16G16_UINT:
            return GpuFormat::RG16_Uint;
        case DXGI_FORMAT_R16G16_SINT:
            return GpuFormat::RG16_Sint;
        case DXGI_FORMAT_R16G16_FLOAT:
            return GpuFormat::RG16_Float;

        case DXGI_FORMAT_R16G16B16A16_UINT:
            return GpuFormat::RGBA16_Uint;
        case DXGI_FORMAT_R16G16B16A16_SINT:
            return GpuFormat::RGBA16_Sint;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return GpuFormat::RGBA16_Float;

        case DXGI_FORMAT_R32_UINT:
            return GpuFormat::R32_Uint;
        case DXGI_FORMAT_R32_SINT:
            return GpuFormat::R32_Sint;
        case DXGI_FORMAT_R32_FLOAT:
            return GpuFormat::R32_Float;

        case DXGI_FORMAT_R32G32_UINT:
            return GpuFormat::RG32_Uint;
        case DXGI_FORMAT_R32G32_SINT:
            return GpuFormat::RG32_Sint;
        case DXGI_FORMAT_R32G32_FLOAT:
            return GpuFormat::RG32_Float;

        case DXGI_FORMAT_R32G32B32_UINT:
            return GpuFormat::RGB32_Uint;
        case DXGI_FORMAT_R32G32B32_SINT:
            return GpuFormat::RGB32_Sint;
        case DXGI_FORMAT_R32G32B32_FLOAT:
            return GpuFormat::RGB32_Float;

        case DXGI_FORMAT_R32G32B32A32_UINT:
            return GpuFormat::RGBA32_Uint;
        case DXGI_FORMAT_R32G32B32A32_SINT:
            return GpuFormat::RGBA32_Sint;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return GpuFormat::RGBA32_Float;

        case DXGI_FORMAT_D16_UNORM:
            return GpuFormat::D16_UNorm;
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
            return GpuFormat::D24_UNorm_S8_Uint;
        case DXGI_FORMAT_D32_FLOAT:
            return GpuFormat::D32_Float;
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
            return GpuFormat::D32_Float_S8X24_Uint;

        default:
            Unreachable("Invalid DXGI format");
        }
    }

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
            Unreachable("Invalid heap type");
        }
    }

    GpuHeap FromD3D12HeapType(D3D12_HEAP_TYPE heap_type)
    {
        switch (heap_type)
        {
        case D3D12_HEAP_TYPE_DEFAULT:
            return GpuHeap::Default;
        case D3D12_HEAP_TYPE_UPLOAD:
            return GpuHeap::Upload;
        case D3D12_HEAP_TYPE_READBACK:
            return GpuHeap::ReadBack;
        default:
            Unreachable("Invalid D3D12 heap type");
        }
    }

    D3D12_RESOURCE_FLAGS ToD3D12ResourceFlags(GpuResourceFlag flags) noexcept
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

    GpuResourceFlag FromD3D12ResourceFlags(D3D12_RESOURCE_FLAGS flags) noexcept
    {
        GpuResourceFlag gpu_flag = GpuResourceFlag::None;
        if (EnumHasAny(flags, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET))
        {
            gpu_flag |= GpuResourceFlag::RenderTarget;
        }
        if (EnumHasAny(flags, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))
        {
            gpu_flag |= GpuResourceFlag::DepthStencil;
        }
        if (EnumHasAny(flags, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
        {
            gpu_flag |= GpuResourceFlag::UnorderedAccess;
        }
        return gpu_flag;
    }

    D3D12_HEAP_FLAGS ToD3D12HeapFlags(GpuResourceFlag flags) noexcept
    {
        D3D12_HEAP_FLAGS heap_flags = D3D12_HEAP_FLAG_NONE;
        if (EnumHasAny(flags, GpuResourceFlag::Shareable))
        {
            heap_flags |= D3D12_HEAP_FLAG_SHARED;
        }
        return heap_flags;
    }

    D3D12_RESOURCE_STATES ToD3D12ResourceState(GpuResourceState state)
    {
        switch (state)
        {
        case GpuResourceState::Common:
            return D3D12_RESOURCE_STATE_COMMON;

        case GpuResourceState::ColorWrite:
            return D3D12_RESOURCE_STATE_RENDER_TARGET;
        case GpuResourceState::DepthWrite:
            return D3D12_RESOURCE_STATE_DEPTH_WRITE;

        case GpuResourceState::UnorderedAccess:
            return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        case GpuResourceState::CopySrc:
            return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case GpuResourceState::CopyDst:
            return D3D12_RESOURCE_STATE_COPY_DEST;

        default:
            Unreachable("Invalid resource state");
        }
    }

    D3D12_RESOURCE_DIMENSION ToD3D12ResourceDimension(GpuResourceType type)
    {
        switch (type)
        {
        case GpuResourceType::Buffer:
            return D3D12_RESOURCE_DIMENSION_BUFFER;
        case GpuResourceType::Texture2D:
        case GpuResourceType::Texture2DArray:
            return D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        case GpuResourceType::Texture3D:
            return D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        default:
            Unreachable("Invalid resource dimension");
        }
    }

    D3D_PRIMITIVE_TOPOLOGY ToD3D12PrimitiveTopology(GpuRenderPipeline::PrimitiveTopology topology) noexcept
    {
        switch (topology)
        {
        case GpuRenderPipeline::PrimitiveTopology::PointList:
            return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
        case GpuRenderPipeline::PrimitiveTopology::TriangleList:
            return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        case GpuRenderPipeline::PrimitiveTopology::TriangleStrip:
            return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
        default:
            Unreachable("Invalid primitive topology");
        }
    }
} // namespace AIHoloImager
