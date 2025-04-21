// Copyright (c) 2024-2005 Minmin Gong
//

#pragma once

#include <cstdint>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif
#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Base/Enum.hpp"
#include "Base/Noncopyable.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuUtil.hpp"

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
        Shareable = 1U << 3,
    };
    ENUM_CLASS_BITWISE_OPERATORS(GpuResourceFlag);

    D3D12_RESOURCE_FLAGS ToD3D12ResourceFlags(GpuResourceFlag flags) noexcept;
    D3D12_HEAP_FLAGS ToD3D12HeapFlags(GpuResourceFlag flags) noexcept;

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

    class GpuSystem;

    class GpuResource
    {
        DISALLOW_COPY_AND_ASSIGN(GpuResource)

    public:
        GpuResource();
        explicit GpuResource(GpuSystem& gpu_system);
        GpuResource(GpuSystem& gpu_system, ID3D12Resource* native_resource);
        virtual ~GpuResource();

        GpuResource(GpuResource&& other) noexcept;
        GpuResource& operator=(GpuResource&& other) noexcept;

        void Name(std::wstring_view name);

        ID3D12Resource* NativeResource() const noexcept;
        explicit operator bool() const noexcept;

        void Reset();

        HANDLE SharedHandle() const noexcept;

    protected:
        void CreateSharedHandle(GpuSystem& gpu_system, GpuResourceFlag flags);

    protected:
        GpuRecyclableObject<ComPtr<ID3D12Resource>> resource_;
        D3D12_RESOURCE_DESC desc_{};
        Win32UniqueHandle shared_handle_;
    };
} // namespace AIHoloImager
