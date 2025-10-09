// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <directx/d3d12.h>

#include "../GpuResourceInternal.hpp"
#include "Base/ComPtr.hpp"
#include "Base/SmartPtrHelper.hpp"
#include "Gpu/GpuResource.hpp"
#include "Gpu/GpuUtil.hpp"

namespace AIHoloImager
{
    class D3D12Resource : public GpuResourceInternal
    {
    public:
        D3D12Resource();
        explicit D3D12Resource(GpuSystem& gpu_system);
        D3D12Resource(GpuSystem& gpu_system, void* native_resource, std::wstring_view name);
        ~D3D12Resource();

        D3D12Resource(D3D12Resource&& other) noexcept;
        D3D12Resource(GpuResourceInternal&& other) noexcept;
        D3D12Resource& operator=(D3D12Resource&& other) noexcept;
        GpuResourceInternal& operator=(GpuResourceInternal&& other) noexcept override;

        void Name(std::wstring_view name) override;

        void* NativeResource() const noexcept override;

        void Reset() override;

        void CreateResource(GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size, uint32_t mip_levels,
            GpuFormat format, GpuHeap heap, GpuResourceFlag flags, GpuResourceState init_state, std::wstring_view name) override;

        void* SharedHandle() const noexcept override;

        GpuResourceType Type() const noexcept override;

        uint32_t Width() const noexcept override;
        uint32_t Height() const noexcept override;
        uint32_t Depth() const noexcept override;
        uint32_t ArraySize() const noexcept override;
        uint32_t MipLevels() const noexcept override;

        GpuFormat Format() const noexcept override;

        GpuResourceFlag Flags() const noexcept override;

    private:
        GpuRecyclableObject<ComPtr<ID3D12Resource>> resource_;
        GpuResourceType type_ = GpuResourceType::Buffer;
        D3D12_RESOURCE_DESC desc_{};
        Win32UniqueHandle shared_handle_;
    };
} // namespace AIHoloImager
