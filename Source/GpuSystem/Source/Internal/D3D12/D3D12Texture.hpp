// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuTexture.hpp"

#include "../GpuTextureInternal.hpp"
#include "D3D12CommandList.hpp"
#include "D3D12ImpDefine.hpp"
#include "D3D12Resource.hpp"

namespace AIHoloImager
{
    class D3D12Texture : public GpuTextureInternal, public D3D12Resource
    {
    public:
        D3D12Texture(GpuSystem& gpu_system, GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size,
            uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::wstring_view name = L"");
        D3D12Texture(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::wstring_view name = L"");
        ~D3D12Texture() noexcept;

        D3D12Texture(D3D12Texture&& other) noexcept;
        explicit D3D12Texture(GpuResourceInternal&& other) noexcept;
        explicit D3D12Texture(GpuTextureInternal&& other) noexcept;
        D3D12Texture& operator=(D3D12Texture&& other) noexcept;
        GpuResourceInternal& operator=(GpuResourceInternal&& other) noexcept override;
        GpuTextureInternal& operator=(GpuTextureInternal&& other) noexcept override;

        void Name(std::wstring_view name) override;

        void* NativeResource() const noexcept override;
        void* NativeTexture() const noexcept override;

        void* SharedHandle() const noexcept override;

        GpuResourceType Type() const noexcept override;
        uint32_t AllocationSize() const noexcept override;

        uint32_t Width(uint32_t mip) const noexcept override;
        uint32_t Height(uint32_t mip) const noexcept override;
        uint32_t Depth(uint32_t mip) const noexcept override;
        uint32_t ArraySize() const noexcept override;
        uint32_t MipLevels() const noexcept override;
        uint32_t Planes() const noexcept override;
        GpuFormat Format() const noexcept override;
        GpuResourceFlag Flags() const noexcept override;

        void Reset() override;

        void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const override;
        void Transition(D3D12CommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        void Transition(D3D12CommandList& cmd_list, GpuResourceState target_state) const override;

    private:
        mutable std::vector<GpuResourceState> curr_states_;
        GpuFormat format_{};
        GpuResourceFlag flags_{};
    };

    D3D12_DEFINE_IMP(Texture)
} // namespace AIHoloImager
