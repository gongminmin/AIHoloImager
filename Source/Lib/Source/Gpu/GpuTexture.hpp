// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <string_view>
#include <vector>

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
#include <directx/dxgiformat.h>

#include "GpuResource.hpp"
#include "GpuUtil.hpp"
#include "Util/ComPtr.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuSystem;
    class GpuCommandList;

    class GpuTexture
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture)

    public:
        GpuTexture();
        GpuTexture(GpuSystem& gpu_system, D3D12_RESOURCE_DIMENSION dim, uint32_t width, uint32_t height, uint32_t depth,
            uint32_t array_size, uint32_t mip_levels, DXGI_FORMAT format, GpuResourceFlag flags, std::wstring_view name = L"");
        GpuTexture(
            GpuSystem& gpu_system, ID3D12Resource* native_resource, GpuResourceState curr_state, std::wstring_view name = L"") noexcept;
        virtual ~GpuTexture() noexcept;

        GpuTexture(GpuTexture&& other) noexcept;
        GpuTexture& operator=(GpuTexture&& other) noexcept;

        ID3D12Resource* NativeTexture() const noexcept;

        explicit operator bool() const noexcept;

        uint32_t Width(uint32_t mip) const noexcept;
        uint32_t Height(uint32_t mip) const noexcept;
        uint32_t Depth(uint32_t mip) const noexcept;
        uint32_t ArraySize() const noexcept;
        uint32_t MipLevels() const noexcept;
        uint32_t Planes() const noexcept;
        DXGI_FORMAT Format() const noexcept;
        D3D12_RESOURCE_FLAGS Flags() const noexcept;

        void Reset();

        void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const;
        void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const;

        void Upload(GpuSystem& gpu_system, GpuCommandList& cmd_list, uint32_t sub_resource, const void* data);
        void Readback(GpuSystem& gpu_system, GpuCommandList& cmd_list, uint32_t sub_resource, void* data) const;
        void CopyFrom(GpuSystem& gpu_system, GpuCommandList& cmd_list, const GpuTexture& other, uint32_t sub_resource, uint32_t dst_x,
            uint32_t dst_y, uint32_t dst_z, const D3D12_BOX& src_box);

    protected:
        GpuRecyclableObject<ComPtr<ID3D12Resource>> resource_;
        D3D12_RESOURCE_DESC desc_{};
        mutable std::vector<D3D12_RESOURCE_STATES> curr_states_;
    };

    class GpuTexture2D final : public GpuTexture
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture2D)

    public:
        GpuTexture2D();
        GpuTexture2D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t mip_levels, DXGI_FORMAT format, GpuResourceFlag flags,
            std::wstring_view name = L"");

        GpuTexture2D(GpuTexture2D&& other) noexcept;
        GpuTexture2D& operator=(GpuTexture2D&& other) noexcept;

        GpuTexture2D Share() const;
    };

    class GpuTexture2DArray final : public GpuTexture
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture2DArray)

    public:
        GpuTexture2DArray();
        GpuTexture2DArray(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t array_size, uint32_t mip_levels,
            DXGI_FORMAT format, GpuResourceFlag flags, std::wstring_view name = L"");

        GpuTexture2DArray(GpuTexture2DArray&& other) noexcept;
        GpuTexture2DArray& operator=(GpuTexture2DArray&& other) noexcept;

        GpuTexture2DArray Share() const;
    };

    class GpuTexture3D final : public GpuTexture
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture3D)

    public:
        GpuTexture3D();
        GpuTexture3D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t depth, uint32_t mip_levels, DXGI_FORMAT format,
            GpuResourceFlag flags, std::wstring_view name = L"");

        GpuTexture3D(GpuTexture3D&& other) noexcept;
        GpuTexture3D& operator=(GpuTexture3D&& other) noexcept;

        GpuTexture3D Share() const;
    };

    uint32_t FormatSize(DXGI_FORMAT fmt) noexcept;
    uint32_t NumPlanes(DXGI_FORMAT fmt) noexcept;
    void DecomposeSubResource(uint32_t sub_resource, uint32_t num_mip_levels, uint32_t array_size, uint32_t& mip_slice,
        uint32_t& array_slice, uint32_t& plane_slice) noexcept;
    uint32_t CalcSubresource(
        uint32_t mip_slice, uint32_t array_slice, uint32_t plane_slice, uint32_t num_mip_levels, uint32_t array_size) noexcept;
} // namespace AIHoloImager
