// Copyright (c) 2024-2025 Minmin Gong
//

#include "Gpu/GpuCommandList.hpp"

#include <cassert>

#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuSystem.hpp"

#include "Internal/GpuSystemInternalFactory.hpp"

namespace AIHoloImager
{
    class GpuCommandList::Impl
    {
    public:
        Impl(GpuSystem& gpu_system, GpuCommandAllocatorInfo& cmd_alloc_info, GpuSystem::CmdQueueType type)
            : gpu_system_(gpu_system), cmd_list_internal_(gpu_system.InternalFactory().CreateCommandList(cmd_alloc_info, type))
        {
        }

        GpuSystem& GpuSys() noexcept
        {
            return gpu_system_;
        }

        GpuCommandListInternal& Internal() noexcept
        {
            return *cmd_list_internal_;
        }

    private:
        GpuSystem& gpu_system_;
        std::unique_ptr<GpuCommandListInternal> cmd_list_internal_;
    };

    GpuCommandList::GpuCommandList() noexcept = default;

    GpuCommandList::GpuCommandList(GpuSystem& gpu_system, GpuCommandAllocatorInfo& cmd_alloc_info, GpuSystem::CmdQueueType type)
        : impl_(std::make_unique<Impl>(gpu_system, cmd_alloc_info, type))
    {
    }

    GpuCommandList::~GpuCommandList() = default;

    GpuCommandList::GpuCommandList(GpuCommandList&& other) noexcept = default;
    GpuCommandList& GpuCommandList::operator=(GpuCommandList&& other) noexcept = default;

    GpuSystem::CmdQueueType GpuCommandList::Type() const noexcept
    {
        assert(impl_);
        return impl_->Internal().Type();
    }

    GpuCommandList::operator bool() const noexcept
    {
        return impl_ && static_cast<bool>(impl_->Internal());
    }

    void* GpuCommandList::NativeCommandListBase() const noexcept
    {
        assert(impl_);
        return impl_->Internal().NativeCommandListBase();
    }

    void GpuCommandList::Clear(GpuRenderTargetView& rtv, const float color[4])
    {
        assert(impl_);
        impl_->Internal().Clear(rtv, color);
    }

    void GpuCommandList::Clear(GpuUnorderedAccessView& uav, const float color[4])
    {
        assert(impl_);
        impl_->Internal().Clear(uav, color);
    }

    void GpuCommandList::Clear(GpuUnorderedAccessView& uav, const uint32_t color[4])
    {
        assert(impl_);
        impl_->Internal().Clear(uav, color);
    }

    void GpuCommandList::ClearDepth(GpuDepthStencilView& dsv, float depth)
    {
        assert(impl_);
        impl_->Internal().ClearDepth(dsv, depth);
    }

    void GpuCommandList::ClearStencil(GpuDepthStencilView& dsv, uint8_t stencil)
    {
        assert(impl_);
        impl_->Internal().ClearStencil(dsv, stencil);
    }

    void GpuCommandList::ClearDepthStencil(GpuDepthStencilView& dsv, float depth, uint8_t stencil)
    {
        assert(impl_);
        impl_->Internal().ClearDepthStencil(dsv, depth, stencil);
    }

    void GpuCommandList::Render(const GpuRenderPipeline& pipeline, std::span<const VertexBufferBinding> vbs, const IndexBufferBinding* ib,
        uint32_t num, std::span<const ShaderBinding> shader_bindings, std::span<const GpuRenderTargetView*> rtvs,
        const GpuDepthStencilView* dsv, std::span<const GpuViewport> viewports, std::span<const GpuRect> scissor_rects)
    {
        assert(impl_);
        impl_->Internal().Render(pipeline, std::move(vbs), ib, num, std::move(shader_bindings), std::move(rtvs), dsv, std::move(viewports),
            std::move(scissor_rects));
    }

    void GpuCommandList::Compute(
        const GpuComputePipeline& pipeline, uint32_t group_x, uint32_t group_y, uint32_t group_z, const ShaderBinding& shader_binding)
    {
        assert(impl_);
        impl_->Internal().Compute(pipeline, group_x, group_y, group_z, shader_binding);
    }

    void GpuCommandList::ComputeIndirect(
        const GpuComputePipeline& pipeline, const GpuBuffer& indirect_args, const ShaderBinding& shader_binding)
    {
        assert(impl_);
        impl_->Internal().ComputeIndirect(pipeline, indirect_args, shader_binding);
    }

    void GpuCommandList::Copy(GpuBuffer& dest, const GpuBuffer& src)
    {
        assert(impl_);
        impl_->Internal().Copy(dest, src);
    }

    void GpuCommandList::Copy(GpuBuffer& dest, uint32_t dst_offset, const GpuBuffer& src, uint32_t src_offset, uint32_t src_size)
    {
        assert(impl_);
        impl_->Internal().Copy(dest, dst_offset, src, src_offset, src_size);
    }

    void GpuCommandList::Copy(GpuTexture& dest, const GpuTexture& src)
    {
        assert(impl_);
        impl_->Internal().Copy(dest, src);
    }

    void GpuCommandList::Copy(GpuTexture& dest, uint32_t dest_sub_resource, uint32_t dst_x, uint32_t dst_y, uint32_t dst_z,
        const GpuTexture& src, uint32_t src_sub_resource, const GpuBox& src_box)
    {
        assert(impl_);
        impl_->Internal().Copy(dest, dest_sub_resource, dst_x, dst_y, dst_z, src, src_sub_resource, src_box);
    }

    void GpuCommandList::Upload(GpuBuffer& dest, const std::function<void(void*)>& copy_func)
    {
        assert(impl_);
        impl_->Internal().Upload(dest, copy_func);
    }

    void GpuCommandList::Upload(GpuBuffer& dest, const void* src_data, uint32_t src_size)
    {
        const uint32_t size = std::min(dest.Size(), src_size);
        this->Upload(dest, [src_data, size](void* dst_data) { std::memcpy(dst_data, src_data, size); });
    }

    void GpuCommandList::Upload(
        GpuTexture& dest, uint32_t sub_resource, const std::function<void(void*, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func)
    {
        assert(impl_);
        impl_->Internal().Upload(dest, sub_resource, copy_func);
    }

    void GpuCommandList::Upload(GpuTexture& dest, uint32_t sub_resource, const void* src_data, uint32_t src_size)
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, dest.MipLevels(), dest.ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = dest.Width(mip);
        const uint32_t height = dest.Height(mip);
        const uint32_t depth = dest.Depth(mip);
        const uint32_t src_row_pitch = width * FormatSize(dest.Format());

        this->Upload(dest, sub_resource,
            [height, depth, src_data, src_size, src_row_pitch](void* dst_data, uint32_t row_pitch, uint32_t slice_pitch) {
                uint32_t size = src_size;
                for (uint32_t z = 0; z < depth; ++z)
                {
                    for (uint32_t y = 0; y < height; ++y)
                    {
                        std::memcpy(&reinterpret_cast<std::byte*>(dst_data)[z * slice_pitch + y * row_pitch],
                            &reinterpret_cast<const std::byte*>(src_data)[(z * height + y) * src_row_pitch], std::min(src_row_pitch, size));
                        size -= src_row_pitch;
                    }
                }
            });
    }

    std::future<void> GpuCommandList::ReadBackAsync(const GpuBuffer& src, const std::function<void(const void*)>& copy_func)
    {
        assert(impl_);
        return impl_->Internal().ReadBackAsync(src, copy_func);
    }

    std::future<void> GpuCommandList::ReadBackAsync(const GpuBuffer& src, void* dst_data, uint32_t dst_size)
    {
        const uint32_t size = std::min(src.Size(), dst_size);
        return this->ReadBackAsync(src, [dst_data, size](const void* src_data) { std::memcpy(dst_data, src_data, size); });
    }

    std::future<void> GpuCommandList::ReadBackAsync(const GpuTexture& src, uint32_t sub_resource,
        const std::function<void(const void*, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func)
    {
        assert(impl_);
        return impl_->Internal().ReadBackAsync(src, sub_resource, copy_func);
    }

    std::future<void> GpuCommandList::ReadBackAsync(const GpuTexture& src, uint32_t sub_resource, void* dst_data, uint32_t dst_size)
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, src.MipLevels(), src.ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = src.Width(mip);
        const uint32_t height = src.Height(mip);
        const uint32_t depth = src.Depth(mip);
        const uint32_t dst_row_pitch = width * FormatSize(src.Format());

        return this->ReadBackAsync(src, sub_resource,
            [height, depth, dst_data, dst_size, dst_row_pitch](const void* src_data, uint32_t row_pitch, uint32_t slice_pitch) {
                uint32_t size = dst_size;
                for (uint32_t z = 0; z < depth; ++z)
                {
                    for (uint32_t y = 0; y < height; ++y)
                    {
                        std::memcpy(&reinterpret_cast<std::byte*>(dst_data)[(z * height + y) * dst_row_pitch],
                            &reinterpret_cast<const std::byte*>(src_data)[z * slice_pitch + y * row_pitch], std::min(dst_row_pitch, size));
                        size -= dst_row_pitch;
                    }
                }
            });
    }

    void GpuCommandList::Close()
    {
        assert(impl_);
        impl_->Internal().Close();
    }

    void GpuCommandList::Reset(GpuCommandAllocatorInfo& cmd_alloc_info)
    {
        assert(impl_);
        impl_->Internal().Reset(cmd_alloc_info);
    }

    GpuCommandAllocatorInfo* GpuCommandList::CommandAllocatorInfo() noexcept
    {
        assert(impl_);
        return impl_->Internal().CommandAllocatorInfo();
    }

    GpuCommandListInternal& GpuCommandList::Internal() noexcept
    {
        assert(impl_);
        return impl_->Internal();
    }

    void GpuCommandList::GenerateMipmaps(GpuTexture2D& texture, GpuSampler::Filter filter)
    {
        auto& mipmapper = impl_->GpuSys().Mipmapper();
        mipmapper.Generate(*this, texture, filter);
    }
} // namespace AIHoloImager
