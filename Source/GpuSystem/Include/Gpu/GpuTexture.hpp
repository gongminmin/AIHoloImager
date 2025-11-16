// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <memory>
#include <string_view>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResource.hpp"
#include "Gpu/InternalDefine.hpp"

namespace AIHoloImager
{
    class GpuSystem;
    class GpuCommandList;

    class GpuTexture : public GpuResource
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture)
        DEFINE_INTERNAL(GpuResource)

    public:
        GpuTexture() noexcept;
        ~GpuTexture() override;

        GpuTexture(GpuTexture&& other) noexcept;
        GpuTexture& operator=(GpuTexture&& other) noexcept;

        void Name(std::string_view name) override;

        void* NativeResource() const noexcept override;
        template <typename Traits>
        typename Traits::ResourceType NativeResource() const noexcept
        {
            return reinterpret_cast<typename Traits::ResourceType>(this->NativeResource());
        }
        void* NativeTexture() const noexcept;
        template <typename Traits>
        typename Traits::TextureType NativeTexture() const noexcept
        {
            return reinterpret_cast<typename Traits::TextureType>(this->NativeTexture());
        }

        explicit operator bool() const noexcept override;

        void* SharedHandle() const noexcept override;

        GpuHeap Heap() const noexcept override;
        GpuResourceType Type() const noexcept override;
        uint32_t AllocationSize() const noexcept override;

        uint32_t Width(uint32_t mip) const noexcept;
        uint32_t Height(uint32_t mip) const noexcept;
        uint32_t Depth(uint32_t mip) const noexcept;
        uint32_t ArraySize() const noexcept;
        uint32_t MipLevels() const noexcept;
        uint32_t Planes() const noexcept;
        GpuFormat Format() const noexcept;
        GpuResourceFlag Flags() const noexcept;

        void Reset();

        void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const override;

    protected:
        GpuTexture(GpuSystem& gpu_system, GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size,
            uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::string_view name = "");
        GpuTexture(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::string_view name = "") noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class GpuTexture2D final : public GpuTexture
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture2D)

    public:
        GpuTexture2D() noexcept;
        GpuTexture2D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags,
            std::string_view name = "");
        GpuTexture2D(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::string_view name = "") noexcept;
        ~GpuTexture2D() override;

        GpuTexture2D(GpuTexture2D&& other) noexcept;
        GpuTexture2D& operator=(GpuTexture2D&& other) noexcept;
    };

    class GpuTexture2DArray final : public GpuTexture
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture2DArray)

    public:
        GpuTexture2DArray() noexcept;
        GpuTexture2DArray(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t array_size, uint32_t mip_levels,
            GpuFormat format, GpuResourceFlag flags, std::string_view name = "");
        GpuTexture2DArray(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::string_view name = "") noexcept;
        ~GpuTexture2DArray() override;

        GpuTexture2DArray(GpuTexture2DArray&& other) noexcept;
        GpuTexture2DArray& operator=(GpuTexture2DArray&& other) noexcept;
    };

    class GpuTexture3D final : public GpuTexture
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture3D)

    public:
        GpuTexture3D() noexcept;
        GpuTexture3D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t depth, uint32_t mip_levels, GpuFormat format,
            GpuResourceFlag flags, std::string_view name = "");
        GpuTexture3D(GpuSystem& gpu_system, void* native_resource, GpuResourceState curr_state, std::string_view name = "") noexcept;
        ~GpuTexture3D() override;

        GpuTexture3D(GpuTexture3D&& other) noexcept;
        GpuTexture3D& operator=(GpuTexture3D&& other) noexcept;
    };

    void DecomposeSubResource(uint32_t sub_resource, uint32_t num_mip_levels, uint32_t array_size, uint32_t& mip_slice,
        uint32_t& array_slice, uint32_t& plane_slice) noexcept;
    uint32_t CalcSubResource(
        uint32_t mip_slice, uint32_t array_slice, uint32_t plane_slice, uint32_t num_mip_levels, uint32_t array_size) noexcept;
} // namespace AIHoloImager
