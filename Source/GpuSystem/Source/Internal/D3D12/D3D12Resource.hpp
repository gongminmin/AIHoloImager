// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Base/MiniWindows.hpp"

#include <directx/d3d12.h>

#include "Base/ComPtr.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResource.hpp"

#include "D3D12Util.hpp"

namespace AIHoloImager
{
    class D3D12Resource
    {
    public:
        explicit D3D12Resource(GpuSystem& gpu_system);
        D3D12Resource(GpuSystem& gpu_system, void* native_resource, std::wstring_view name);
        virtual ~D3D12Resource();

        D3D12Resource(D3D12Resource&& other) noexcept;
        D3D12Resource& operator=(D3D12Resource&& other) noexcept;

        void Name(std::wstring_view name);

        void* NativeResource() const noexcept;

        void Reset();

        void CreateResource(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size, uint32_t mip_levels,
            GpuFormat format, GpuHeap heap, GpuResourceFlag flags, GpuResourceState init_state, std::wstring_view name);

        void* SharedHandle() const noexcept;

        GpuResourceType Type() const noexcept;

        uint32_t Width() const noexcept;
        uint32_t Height() const noexcept;
        uint32_t Depth() const noexcept;
        uint32_t ArraySize() const noexcept;
        uint32_t MipLevels() const noexcept;

        GpuFormat Format() const noexcept;

        GpuResourceFlag Flags() const noexcept;

    private:
        D3D12RecyclableObject<ComPtr<ID3D12Resource>> resource_;
        GpuResourceType type_ = GpuResourceType::Buffer;
        D3D12_RESOURCE_DESC desc_{};
        Win32UniqueHandle shared_handle_;
    };
} // namespace AIHoloImager
