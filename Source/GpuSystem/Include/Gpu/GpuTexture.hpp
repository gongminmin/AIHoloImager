// Copyright (c) 2024-2026 Minmin Gong
//

#pragma once

#include <memory>
#include <string_view>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuFormat.hpp"
#include "Gpu/GpuResource.hpp"
#include "Gpu/InternalDefine.hpp"
#include "Gpu/Symbol.hpp"

namespace AIHoloImager
{
    class GpuSystem;
    class GpuCommandList;

    class GpuTexture : public GpuResource
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture)
        DEFINE_INTERNAL(GpuResource)

    public:
        AIHI_GPU_SYS_API GpuTexture() noexcept;
        AIHI_GPU_SYS_API ~GpuTexture() override;

        AIHI_GPU_SYS_API GpuTexture(GpuTexture&& other) noexcept;
        AIHI_GPU_SYS_API GpuTexture& operator=(GpuTexture&& other) noexcept;

        AIHI_GPU_SYS_API void Name(std::string_view name) override;

        AIHI_GPU_SYS_API void* NativeResource() const noexcept override;
        template <typename Traits>
        typename Traits::ResourceType NativeResource() const noexcept
        {
            return reinterpret_cast<typename Traits::ResourceType>(this->NativeResource());
        }
        AIHI_GPU_SYS_API void* NativeTexture() const noexcept;
        template <typename Traits>
        typename Traits::TextureType NativeTexture() const noexcept
        {
            return reinterpret_cast<typename Traits::TextureType>(this->NativeTexture());
        }

        AIHI_GPU_SYS_API explicit operator bool() const noexcept override;

        AIHI_GPU_SYS_API void* SharedHandle() const noexcept override;

        AIHI_GPU_SYS_API GpuHeap Heap() const noexcept override;
        AIHI_GPU_SYS_API GpuResourceType Type() const noexcept override;
        AIHI_GPU_SYS_API GpuResourceFlag Flags() const noexcept override;
        AIHI_GPU_SYS_API uint32_t AllocationSize() const noexcept override;

        AIHI_GPU_SYS_API uint32_t Width(uint32_t mip) const noexcept;
        AIHI_GPU_SYS_API uint32_t Height(uint32_t mip) const noexcept;
        AIHI_GPU_SYS_API uint32_t Depth(uint32_t mip) const noexcept;
        AIHI_GPU_SYS_API uint32_t ArraySize() const noexcept;
        AIHI_GPU_SYS_API uint32_t MipLevels() const noexcept;
        AIHI_GPU_SYS_API uint32_t Planes() const noexcept;
        AIHI_GPU_SYS_API GpuFormat Format() const noexcept;

        AIHI_GPU_SYS_API void Reset();

        AIHI_GPU_SYS_API void Transition(GpuCommandList& cmd_list, uint32_t sub_resource, GpuResourceState target_state) const override;
        AIHI_GPU_SYS_API void Transition(GpuCommandList& cmd_list, GpuResourceState target_state) const override;

        AIHI_GPU_SYS_API const std::shared_ptr<GpuSystem::WaitFences>& StalledWaitFences() const noexcept override;

    protected:
        GpuTexture(GpuSystem& gpu_system, GpuResourceType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t array_size,
            uint32_t mip_levels, GpuFormat format, GpuResourceFlag flags, std::string_view name = "");

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class GpuTexture2D final : public GpuTexture
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture2D)

    public:
        AIHI_GPU_SYS_API GpuTexture2D() noexcept;
        AIHI_GPU_SYS_API GpuTexture2D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t mip_levels, GpuFormat format,
            GpuResourceFlag flags, std::string_view name = "");
        AIHI_GPU_SYS_API ~GpuTexture2D() override;

        AIHI_GPU_SYS_API GpuTexture2D(GpuTexture2D&& other) noexcept;
        AIHI_GPU_SYS_API GpuTexture2D& operator=(GpuTexture2D&& other) noexcept;
    };

    class GpuTexture2DArray final : public GpuTexture
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture2DArray)

    public:
        AIHI_GPU_SYS_API GpuTexture2DArray() noexcept;
        AIHI_GPU_SYS_API GpuTexture2DArray(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t array_size, uint32_t mip_levels,
            GpuFormat format, GpuResourceFlag flags, std::string_view name = "");
        AIHI_GPU_SYS_API ~GpuTexture2DArray() override;

        AIHI_GPU_SYS_API GpuTexture2DArray(GpuTexture2DArray&& other) noexcept;
        AIHI_GPU_SYS_API GpuTexture2DArray& operator=(GpuTexture2DArray&& other) noexcept;
    };

    class GpuTexture3D final : public GpuTexture
    {
        DISALLOW_COPY_AND_ASSIGN(GpuTexture3D)

    public:
        AIHI_GPU_SYS_API GpuTexture3D() noexcept;
        AIHI_GPU_SYS_API GpuTexture3D(GpuSystem& gpu_system, uint32_t width, uint32_t height, uint32_t depth, uint32_t mip_levels,
            GpuFormat format, GpuResourceFlag flags, std::string_view name = "");
        AIHI_GPU_SYS_API ~GpuTexture3D() override;

        AIHI_GPU_SYS_API GpuTexture3D(GpuTexture3D&& other) noexcept;
        AIHI_GPU_SYS_API GpuTexture3D& operator=(GpuTexture3D&& other) noexcept;
    };

    AIHI_GPU_SYS_API void DecomposeSubResource(uint32_t sub_resource, uint32_t num_mip_levels, uint32_t array_size, uint32_t& mip_slice,
        uint32_t& array_slice, uint32_t& plane_slice) noexcept;
    AIHI_GPU_SYS_API uint32_t CalcSubResource(
        uint32_t mip_slice, uint32_t array_slice, uint32_t plane_slice, uint32_t num_mip_levels, uint32_t array_size) noexcept;
} // namespace AIHoloImager
