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

#include "Util/ComPtr.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class GpuSystem;
    class GpuCommandList;

    class GpuTexture2D final
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture2D)

    public:
        GpuTexture2D();
        GpuTexture2D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t mip_levels, DXGI_FORMAT format,
            D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES init_state, std::wstring_view name = L"");
        GpuTexture2D(ID3D12Resource* native_resource, D3D12_RESOURCE_STATES curr_state, std::wstring_view name = L"") noexcept;
        ~GpuTexture2D() noexcept;

        GpuTexture2D(GpuTexture2D&& other) noexcept;
        GpuTexture2D& operator=(GpuTexture2D&& other) noexcept;

        GpuTexture2D Share() const;

        ID3D12Resource* NativeTexture() const noexcept;

        explicit operator bool() const noexcept;

        uint32_t Width(uint32_t mip) const noexcept;
        uint32_t Height(uint32_t mip) const noexcept;
        uint32_t MipLevels() const noexcept;
        uint32_t Planes() const noexcept;
        DXGI_FORMAT Format() const noexcept;
        D3D12_RESOURCE_FLAGS Flags() const noexcept;

        void Reset() noexcept;

        D3D12_RESOURCE_STATES State(uint32_t sub_resource) const noexcept;
        void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, D3D12_RESOURCE_STATES target_state) const;
        void Transition(GpuCommandList& cmd_list, D3D12_RESOURCE_STATES target_state) const;

        void Upload(GpuSystem& gpu_system, GpuCommandList& cmd_list, uint32_t sub_resource, const void* data);
        void Readback(GpuSystem& gpu_system, GpuCommandList& cmd_list, uint32_t sub_resource, void* data) const;
        void CopyFrom(GpuSystem& gpu_system, GpuCommandList& cmd_list, const GpuTexture2D& other, uint32_t sub_resource, uint32_t dst_x,
            uint32_t dst_y, const D3D12_BOX& src_box);

    private:
        ComPtr<ID3D12Resource> resource_;
        D3D12_RESOURCE_DESC desc_{};
        mutable std::vector<D3D12_RESOURCE_STATES> curr_states_;
    };

    uint32_t FormatSize(DXGI_FORMAT fmt) noexcept;
    uint32_t NumPlanes(DXGI_FORMAT fmt) noexcept;
    void SubResourceToMipLevelPlane(uint32_t sub_resource, uint32_t num_mip_levels, uint32_t& mip, uint32_t& plane);
} // namespace AIHoloImager
