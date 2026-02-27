// Copyright (c) 2025-2026 Minmin Gong
//

#include "VulkanCommandList.hpp"

#include <bit>
#include <cassert>
#include <format>
#include <iostream>

#include "Base/ErrorHandling.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuSystem.hpp"

#include "VulkanBuffer.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanErrorhandling.hpp"
#include "VulkanResource.hpp"
#include "VulkanResourceViews.hpp"
#include "VulkanSampler.hpp"
#include "VulkanShader.hpp"
#include "VulkanSystem.hpp"
#include "VulkanTexture.hpp"

namespace
{
    constexpr const char* YellowEscape = "\033[33m";
    constexpr const char* EndEscape = "\033[0m";
} // namespace

namespace AIHoloImager
{
    VULKAN_IMP_IMP(CommandList)

    VulkanCommandList::VulkanCommandList(GpuSystem& gpu_system, GpuCommandPool& cmd_pool, GpuSystem::CmdQueueType type)
        : gpu_system_(&gpu_system), type_(type)
    {
        this->Reset(cmd_pool);
    }

    VulkanCommandList::~VulkanCommandList()
    {
        if ((cmd_buff_ != VK_NULL_HANDLE) && !closed_)
        {
            Unreachable("Command list is destructed without executing.");
        }
    }

    VulkanCommandList::VulkanCommandList(VulkanCommandList&& other) noexcept = default;
    VulkanCommandList::VulkanCommandList(GpuCommandListInternal&& other) noexcept
        : VulkanCommandList(static_cast<VulkanCommandList&&>(other))
    {
    }

    VulkanCommandList& VulkanCommandList::operator=(VulkanCommandList&& other) noexcept = default;
    GpuCommandListInternal& VulkanCommandList::operator=(GpuCommandListInternal&& other) noexcept
    {
        return this->operator=(static_cast<GpuCommandListInternal&&>(other));
    }

    GpuSystem::CmdQueueType VulkanCommandList::Type() const noexcept
    {
        return type_;
    }

    VulkanCommandList::operator bool() const noexcept
    {
        return cmd_buff_ != VK_NULL_HANDLE;
    }

    void VulkanCommandList::Transition(std::span<const VkBufferMemoryBarrier> barriers) const noexcept
    {
        vkCmdPipelineBarrier(cmd_buff_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
            static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
    }

    void VulkanCommandList::Transition(std::span<const VkImageMemoryBarrier> barriers) const noexcept
    {
        vkCmdPipelineBarrier(cmd_buff_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
            static_cast<uint32_t>(barriers.size()), barriers.data());
    }

    void VulkanCommandList::Clear(GpuRenderTargetView& rtv, const float color[4])
    {
        auto& vulkan_rtv = VulkanImp(rtv);

        const VkClearColorValue clear_color{
            .float32 = {color[0], color[1], color[2], color[3]},
        };
        const auto& subres_range = vulkan_rtv.SubresourceRange();
        VulkanImp(*vulkan_rtv.Resource()).Transition(*this, GpuResourceState::UnorderedAccess);

        vkCmdClearColorImage(cmd_buff_, vulkan_rtv.Image(), VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &subres_range);
    }

    void VulkanCommandList::Clear(GpuUnorderedAccessView& uav, const float color[4])
    {
        auto& vulkan_uav = VulkanImp(uav);
        auto& vulkan_resource = VulkanImp(*vulkan_uav.Resource());
        if (vulkan_uav.BufferView() != VK_NULL_HANDLE)
        {
            const auto& [buff_offset, buffer_size] = vulkan_uav.BufferRange();
            vulkan_resource.Transition(*this, GpuResourceState::UnorderedAccess);

            vkCmdFillBuffer(cmd_buff_, vulkan_uav.Buffer(), buff_offset, buffer_size, std::bit_cast<uint32_t>(color[0]));
        }
        else
        {
            const VkClearColorValue clear_color{
                .float32 = {color[0], color[1], color[2], color[3]},
            };
            const auto& subres_range = vulkan_uav.SubresourceRange();
            vulkan_resource.Transition(*this, GpuResourceState::UnorderedAccess);

            vkCmdClearColorImage(cmd_buff_, vulkan_uav.Image(), VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &subres_range);
        }
    }

    void VulkanCommandList::Clear(GpuUnorderedAccessView& uav, const uint32_t color[4])
    {
        auto& vulkan_uav = VulkanImp(uav);
        auto& vulkan_resource = VulkanImp(*vulkan_uav.Resource());
        if (vulkan_uav.BufferView() != VK_NULL_HANDLE)
        {
            const auto& [buff_offset, buffer_size] = vulkan_uav.BufferRange();
            vulkan_resource.Transition(*this, GpuResourceState::UnorderedAccess);

            vkCmdFillBuffer(cmd_buff_, vulkan_uav.Buffer(), buff_offset, buffer_size, color[0]);
        }
        else
        {
            const VkClearColorValue clear_color{
                .uint32 = {color[0], color[1], color[2], color[3]},
            };
            const auto& subres_range = vulkan_uav.SubresourceRange();
            vulkan_resource.Transition(*this, GpuResourceState::UnorderedAccess);

            vkCmdClearColorImage(cmd_buff_, vulkan_uav.Image(), VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &subres_range);
        }
    }

    void VulkanCommandList::ClearDepth(GpuDepthStencilView& dsv, float depth)
    {
        auto& vulkan_dsv = VulkanImp(dsv);

        const VkClearDepthStencilValue clear_depth{
            .depth = depth,
        };
        VkImageSubresourceRange subres_range = vulkan_dsv.SubresourceRange();
        subres_range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        VulkanImp(*vulkan_dsv.Resource()).Transition(*this, GpuResourceState::UnorderedAccess);

        vkCmdClearDepthStencilImage(cmd_buff_, vulkan_dsv.Image(), VK_IMAGE_LAYOUT_GENERAL, &clear_depth, 1, &subres_range);
    }

    void VulkanCommandList::ClearStencil(GpuDepthStencilView& dsv, uint8_t stencil)
    {
        auto& vulkan_dsv = VulkanImp(dsv);

        const VkClearDepthStencilValue clear_stencil{
            .stencil = stencil,
        };
        VkImageSubresourceRange subres_range = vulkan_dsv.SubresourceRange();
        subres_range.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        VulkanImp(*vulkan_dsv.Resource()).Transition(*this, GpuResourceState::UnorderedAccess);

        vkCmdClearDepthStencilImage(cmd_buff_, vulkan_dsv.Image(), VK_IMAGE_LAYOUT_GENERAL, &clear_stencil, 1, &subres_range);
    }

    void VulkanCommandList::ClearDepthStencil(GpuDepthStencilView& dsv, float depth, uint8_t stencil)
    {
        auto& vulkan_dsv = VulkanImp(dsv);

        const VkClearDepthStencilValue clear_depth_stencil{
            .depth = depth,
            .stencil = stencil,
        };
        VkImageSubresourceRange subres_range = vulkan_dsv.SubresourceRange();
        subres_range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        VulkanImp(*vulkan_dsv.Resource()).Transition(*this, GpuResourceState::UnorderedAccess);

        vkCmdClearDepthStencilImage(cmd_buff_, vulkan_dsv.Image(), VK_IMAGE_LAYOUT_GENERAL, &clear_depth_stencil, 1, &subres_range);
    }

    void VulkanCommandList::Render(const GpuRenderPipeline& pipeline, std::span<const GpuCommandList::VertexBufferBinding> vbs,
        const GpuCommandList::IndexBufferBinding* ib, uint32_t num, std::span<const GpuCommandList::ShaderBinding> shader_bindings,
        std::span<GpuRenderTargetView*> rtvs, GpuDepthStencilView* dsv, std::span<const GpuViewport> viewports,
        std::span<const GpuRect> scissor_rects)
    {
        assert(gpu_system_ != nullptr);

        auto& vulkan_system = VulkanImp(*gpu_system_);
        const VkDevice vulkan_device = vulkan_system.Device();

        const auto& vulkan_pipeline = VulkanImp(pipeline);

        std::unique_ptr<VkBuffer[]> vbvs;
        std::unique_ptr<VkDeviceSize[]> vb_offsets;
        if (!vbs.empty())
        {
            vbvs = std::make_unique<VkBuffer[]>(vbs.size());
            vb_offsets = std::make_unique<VkDeviceSize[]>(vbs.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(vbs.size()); ++i)
            {
                const auto& vb_binding = vbs[i];
                assert(vb_binding.vb != nullptr);

                const auto& vulkan_buff = VulkanImp(*vb_binding.vb);
                vulkan_buff.Transition(*this, GpuResourceState::Common);

                vbvs[i] = vulkan_buff.Buffer();
                vb_offsets[i] = vb_binding.offset;
            }
        }

        VkBuffer ibv = VK_NULL_HANDLE;
        VkDeviceSize ib_offset = 0;
        VkIndexType index_type = VK_INDEX_TYPE_UINT16;
        if (ib != nullptr)
        {
            const auto& vulkan_buff = VulkanImp(*ib->ib);
            vulkan_buff.Transition(*this, GpuResourceState::Common);

            ibv = vulkan_buff.Buffer();
            ib_offset = ib->offset;

            switch (ib->format)
            {
            case GpuFormat::R16_Uint:
                index_type = VK_INDEX_TYPE_UINT16;
                break;
            case GpuFormat::R32_Uint:
                index_type = VK_INDEX_TYPE_UINT32;
                break;

            default:
                Unreachable("Unsupported index format");
            }
        }

        VkDescriptorSet desc_sets[] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        const uint32_t num_sets = vulkan_pipeline.NumDescSetLayouts();
        for (uint32_t set = 0; set < num_sets; ++set)
        {
            desc_sets[set] = vulkan_system.AllocDescSet(vulkan_pipeline.DescSetLayout(set));
        }

        std::vector<VkWriteDescriptorSet> write_desc_sets;
        for (uint32_t s = 0; s < static_cast<size_t>(GpuRenderPipeline::ShaderStage::Num); ++s)
        {
            const auto stage = static_cast<GpuRenderPipeline::ShaderStage>(s);
            const auto& binding_slots = vulkan_pipeline.BindingSlots(stage);
            const auto& shader_name = vulkan_pipeline.ShaderName(stage);
            const auto& shader_binding = shader_bindings[s];

            this->GenWriteDescSet(write_desc_sets, binding_slots, shader_name, shader_binding, desc_sets);
        }

        if (!write_desc_sets.empty())
        {
            vkUpdateDescriptorSets(vulkan_device, static_cast<uint32_t>(write_desc_sets.size()), write_desc_sets.data(), 0, nullptr);
        }

        uint32_t render_width = 0;
        uint32_t render_height = 0;

        std::vector<VkRenderingAttachmentInfo> color_attachments(rtvs.size());
        for (size_t i = 0; i < rtvs.size(); ++i)
        {
            auto& vulkan_rtv = VulkanImp(*rtvs[i]);
            vulkan_rtv.Transition(*this);

            color_attachments[i] = VkRenderingAttachmentInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = vulkan_rtv.ImageView(),
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            };

            const auto& vulkan_texture = static_cast<VulkanTexture&>(VulkanImp(*vulkan_rtv.Resource()));
            const uint32_t mip = vulkan_rtv.SubresourceRange().baseMipLevel;

            render_width = std::max(render_width, vulkan_texture.Width(mip));
            render_height = std::max(render_height, vulkan_texture.Height(mip));
        }
        VkRenderingAttachmentInfo depth_stencil_attachment;
        bool has_stencil = false;
        if (dsv != nullptr)
        {
            auto& vulkan_dsv = VulkanImp(*dsv);
            vulkan_dsv.Transition(*this);

            depth_stencil_attachment = VkRenderingAttachmentInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = vulkan_dsv.ImageView(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            };

            const auto& vulkan_texture = static_cast<VulkanTexture&>(VulkanImp(*vulkan_dsv.Resource()));
            const uint32_t mip = vulkan_dsv.SubresourceRange().baseMipLevel;

            render_width = std::max(render_width, vulkan_texture.Width(mip));
            render_height = std::max(render_height, vulkan_texture.Height(mip));

            has_stencil = IsStencilFormat(vulkan_texture.Format());
        }

        const VkRenderingInfo rendering_info{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea{0, 0, render_width, render_height},
            .layerCount = 1,
            .colorAttachmentCount = static_cast<uint32_t>(color_attachments.size()),
            .pColorAttachments = color_attachments.data(),
            .pDepthAttachment = dsv != nullptr ? &depth_stencil_attachment : nullptr,
            .pStencilAttachment = has_stencil ? &depth_stencil_attachment : nullptr,
        };
        vkCmdBeginRendering(cmd_buff_, &rendering_info);

        if (!vbs.empty())
        {
            vkCmdBindVertexBuffers(cmd_buff_, 0, static_cast<uint32_t>(vbs.size()), vbvs.get(), vb_offsets.get());
        }
        if (ib != nullptr)
        {
            vkCmdBindIndexBuffer(cmd_buff_, ibv, ib_offset, index_type);
        }

        vulkan_pipeline.Bind(*this);

        if (num_sets > 0)
        {
            vkCmdBindDescriptorSets(
                cmd_buff_, VK_PIPELINE_BIND_POINT_GRAPHICS, vulkan_pipeline.PipelineLayout(), 0, num_sets, desc_sets, 0, 0);
        }

        auto vulkan_viewports = std::make_unique<VkViewport[]>(viewports.size());
        for (size_t i = 0; i < viewports.size(); ++i)
        {
            vulkan_viewports[i] = VkViewport{
                .x = viewports[i].left,
                .y = viewports[i].top,
                .width = viewports[i].width,
                .height = viewports[i].height,
                .minDepth = viewports[i].min_depth,
                .maxDepth = viewports[i].max_depth,
            };
        }
        vkCmdSetViewport(cmd_buff_, 0, static_cast<uint32_t>(viewports.size()), vulkan_viewports.get());

        if (scissor_rects.empty())
        {
            const VkRect2D vulkan_scissor_rect{
                .offset{
                    .x = std::max(0, static_cast<int32_t>(viewports[0].left)),
                    .y = std::max(0, static_cast<int32_t>(viewports[0].top)),
                },
                .extent{
                    .width = std::min(render_width, static_cast<uint32_t>(viewports[0].width)),
                    .height = std::min(render_height, static_cast<uint32_t>(viewports[0].height)),
                },
            };
            vkCmdSetScissor(cmd_buff_, 0, 1, &vulkan_scissor_rect);
        }
        else
        {
            auto vulkan_scissor_rects = std::make_unique<VkRect2D[]>(scissor_rects.size());
            for (size_t i = 0; i < scissor_rects.size(); ++i)
            {
                vulkan_scissor_rects[i] = {
                    .offset{
                        .x = static_cast<int32_t>(scissor_rects[i].left),
                        .y = static_cast<int32_t>(scissor_rects[i].top),
                    },
                    .extent{
                        .width = static_cast<uint32_t>(scissor_rects[i].right - scissor_rects[i].left),
                        .height = static_cast<uint32_t>(scissor_rects[i].bottom - scissor_rects[i].top),
                    },
                };
            }
            vkCmdSetScissor(cmd_buff_, 0, static_cast<uint32_t>(scissor_rects.size()), vulkan_scissor_rects.get());
        }

        if (ib != nullptr)
        {
            vkCmdDrawIndexed(cmd_buff_, num, 1, 0, 0, 0);
        }
        else
        {
            vkCmdDraw(cmd_buff_, num, 1, 0, 0);
        }

        vkCmdEndRendering(cmd_buff_);

        for (uint32_t set = 0; set < num_sets; ++set)
        {
            vulkan_system.DeallocDescSet(desc_sets[set]);
        }
    }

    void VulkanCommandList::Compute(const GpuComputePipeline& pipeline, uint32_t group_x, uint32_t group_y, uint32_t group_z,
        const GpuCommandList::ShaderBinding& shader_binding)
    {
        this->Compute(
            pipeline, shader_binding, [this, group_x, group_y, group_z]() { vkCmdDispatch(cmd_buff_, group_x, group_y, group_z); });
    }

    void VulkanCommandList::ComputeIndirect(
        const GpuComputePipeline& pipeline, const GpuBuffer& indirect_args, const GpuCommandList::ShaderBinding& shader_binding)
    {
        this->Compute(pipeline, shader_binding, [this, &indirect_args]() {
            const auto& vulkan_indirect_args = VulkanImp(indirect_args);
            vulkan_indirect_args.Transition(*this, GpuResourceState::Common);

            vkCmdDispatchIndirect(cmd_buff_, vulkan_indirect_args.Buffer(), 0);
            ;
        });
    }

    void VulkanCommandList::Compute(
        const GpuComputePipeline& pipeline, const GpuCommandList::ShaderBinding& shader_binding, std::function<void()> dispatch_call)
    {
        assert(gpu_system_ != nullptr);

        auto& vulkan_system = VulkanImp(*gpu_system_);
        const VkDevice vulkan_device = vulkan_system.Device();

        const auto& vulkan_pipeline = VulkanImp(pipeline);
        vulkan_pipeline.Bind(*this);

        const auto& binding_slots = vulkan_pipeline.BindingSlots();
        const auto& shader_name = vulkan_pipeline.ShaderName();

        VkDescriptorSet desc_sets[] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        const uint32_t num_sets = vulkan_pipeline.NumDescSetLayouts();
        for (uint32_t set = 0; set < num_sets; ++set)
        {
            desc_sets[set] = vulkan_system.AllocDescSet(vulkan_pipeline.DescSetLayout(set));
        }

        std::vector<VkWriteDescriptorSet> write_desc_sets;
        this->GenWriteDescSet(write_desc_sets, binding_slots, shader_name, shader_binding, desc_sets);

        if (!write_desc_sets.empty())
        {
            vkUpdateDescriptorSets(vulkan_device, static_cast<uint32_t>(write_desc_sets.size()), write_desc_sets.data(), 0, nullptr);
        }
        vkCmdBindDescriptorSets(cmd_buff_, VK_PIPELINE_BIND_POINT_COMPUTE, vulkan_pipeline.PipelineLayout(), 0, num_sets, desc_sets, 0, 0);

        dispatch_call();

        for (uint32_t set = 0; set < num_sets; ++set)
        {
            vulkan_system.DeallocDescSet(desc_sets[set]);
        }
    }

    void VulkanCommandList::Copy(GpuBuffer& dest, const GpuBuffer& src)
    {
        const auto& vulkan_src = VulkanImp(src);
        auto& vulkan_dst = VulkanImp(dest);

        vulkan_src.Transition(*this, GpuResourceState::CopySrc);
        vulkan_dst.Transition(*this, GpuResourceState::CopyDst);

        const VkBufferCopy2 copy_region{
            .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            .srcOffset = 0,
            .dstOffset = 0,
            .size = src.Size(),
        };
        const VkCopyBufferInfo2 copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            .srcBuffer = vulkan_src.Buffer(),
            .dstBuffer = vulkan_dst.Buffer(),
            .regionCount = 1,
            .pRegions = &copy_region,
        };
        vkCmdCopyBuffer2(cmd_buff_, &copy_info);
    }

    void VulkanCommandList::Copy(GpuBuffer& dest, uint32_t dst_offset, const GpuBuffer& src, uint32_t src_offset, uint32_t src_size)
    {
        const auto& vulkan_src = VulkanImp(src);
        auto& vulkan_dst = VulkanImp(dest);

        vulkan_src.Transition(*this, GpuResourceState::CopySrc);
        vulkan_dst.Transition(*this, GpuResourceState::CopyDst);

        const VkBufferCopy2 copy_region{
            .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            .srcOffset = src_offset,
            .dstOffset = dst_offset,
            .size = src_size,
        };
        const VkCopyBufferInfo2 copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            .srcBuffer = vulkan_src.Buffer(),
            .dstBuffer = vulkan_dst.Buffer(),
            .regionCount = 1,
            .pRegions = &copy_region,
        };
        vkCmdCopyBuffer2(cmd_buff_, &copy_info);
    }

    void VulkanCommandList::Copy(GpuTexture& dest, const GpuTexture& src)
    {
        const auto& vulkan_src = VulkanImp(src);
        auto& vulkan_dst = VulkanImp(dest);

        vulkan_src.Transition(*this, GpuResourceState::CopySrc);
        vulkan_dst.Transition(*this, GpuResourceState::CopyDst);

        const VkCopyImageInfo2 copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_IMAGE_INFO,
            .srcImage = vulkan_src.Image(),
            .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .dstImage = vulkan_dst.Image(),
            .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        };
        vkCmdCopyImage2(cmd_buff_, &copy_info);
    }

    void VulkanCommandList::Copy(GpuTexture& dest, uint32_t dest_sub_resource, uint32_t dst_x, uint32_t dst_y, uint32_t dst_z,
        const GpuTexture& src, uint32_t src_sub_resource, const GpuBox& src_box)
    {
        const auto& vulkan_src = VulkanImp(src);
        auto& vulkan_dst = VulkanImp(dest);

        vulkan_src.Transition(*this, GpuResourceState::CopySrc);
        vulkan_dst.Transition(*this, GpuResourceState::CopyDst);

        uint32_t src_mip;
        uint32_t src_array_slice;
        uint32_t src_plane_slice;
        DecomposeSubResource(src_sub_resource, src.MipLevels(), src.ArraySize(), src_mip, src_array_slice, src_plane_slice);

        uint32_t dst_mip;
        uint32_t dst_array_slice;
        uint32_t dst_plane_slice;
        DecomposeSubResource(dest_sub_resource, dest.MipLevels(), dest.ArraySize(), dst_mip, dst_array_slice, dst_plane_slice);

        const VkImageCopy2 copy_region{
            .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
            .srcSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = src_mip,
                .baseArrayLayer = src_array_slice,
                .layerCount = 1,
            },
            .srcOffset{
                .x = static_cast<int32_t>(src_box.left),
                .y = static_cast<int32_t>(src_box.top),
                .z = static_cast<int32_t>(src_box.front),
            },
            .dstSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = dst_mip,
                .baseArrayLayer = dst_array_slice,
                .layerCount = 1,
            },
            .dstOffset{
                .x = static_cast<int32_t>(dst_x),
                .y = static_cast<int32_t>(dst_y),
                .z = static_cast<int32_t>(dst_z),
            },
            .extent{
                .width = src_box.right - src_box.left,
                .height = src_box.bottom - src_box.top,
                .depth = src_box.back - src_box.front,
            },
        };
        const VkCopyImageInfo2 copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
            .srcImage = vulkan_src.Image(),
            .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .dstImage = vulkan_dst.Image(),
            .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .regionCount = 1,
            .pRegions = &copy_region,
        };
        vkCmdCopyImage2(cmd_buff_, &copy_info);
    }

    void VulkanCommandList::Upload(GpuBuffer& dest, const std::function<void(void* dst_data)>& copy_func)
    {
        auto& vulkan_dst = VulkanImp(dest);

        switch (dest.Heap())
        {
        case GpuHeap::Upload:
        case GpuHeap::ReadBack:
            copy_func(dest.Map());
            dest.Unmap();
            break;

        case GpuHeap::Default:
        {
            auto upload_mem_block = gpu_system_->AllocUploadMemBlock(dest.Size(), gpu_system_->StructuredDataAlignment());

            void* buff_data = upload_mem_block.CpuSpan<std::byte>().data();
            copy_func(buff_data);

            vulkan_dst.Transition(*this, GpuResourceState::CopyDst);

            const VkBufferCopy2 copy_region{
                .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                .srcOffset = upload_mem_block.Offset(),
                .dstOffset = 0,
                .size = dest.Size(),
            };
            const VkCopyBufferInfo2 copy_info{
                .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                .srcBuffer = VulkanImp(*upload_mem_block.Buffer()).Buffer(),
                .dstBuffer = vulkan_dst.Buffer(),
                .regionCount = 1,
                .pRegions = &copy_region,
            };
            vkCmdCopyBuffer2(cmd_buff_, &copy_info);

            gpu_system_->DeallocUploadMemBlock(std::move(upload_mem_block));
        }
        break;

        default:
            Unreachable("Invalid heap type");
        }
    }

    void VulkanCommandList::Upload(GpuTexture& dest, uint32_t sub_resource,
        const std::function<void(void* dst_data, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func)
    {
        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, dest.MipLevels(), dest.ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = dest.Width(mip);
        const uint32_t height = dest.Height(mip);
        const uint32_t depth = dest.Depth(mip);

        const VkDevice vulkan_device = VulkanImp(*gpu_system_).Device();
        auto& vulkan_dst = VulkanImp(dest);
        const VkImage vulkan_dst_image = vulkan_dst.Image();

        VkMemoryRequirements requirements;
        vkGetImageMemoryRequirements(vulkan_device, vulkan_dst_image, &requirements);

        auto upload_mem_block =
            gpu_system_->AllocUploadMemBlock(static_cast<uint32_t>(requirements.size), gpu_system_->TextureDataAlignment());

        VkBuffer vulkan_src_buffer = VulkanImp(*upload_mem_block.Buffer()).Buffer();

        void* tex_data = upload_mem_block.CpuSpan<std::byte>().data();
        const uint32_t row_pitch = dest.Width(mip) * FormatSize(dest.Format());
        copy_func(tex_data, row_pitch, row_pitch * dest.Height(mip));

        vulkan_dst.Transition(*this, GpuResourceState::CopyDst);

        const VkBufferImageCopy2 region{
            .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .bufferOffset = upload_mem_block.Offset(),
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = mip,
                .baseArrayLayer = array_slice,
                .layerCount = 1,
            },
            .imageOffset{
                .x = 0,
                .y = 0,
                .z = 0,
            },
            .imageExtent{
                .width = width,
                .height = height,
                .depth = depth,
            },
        };
        const VkCopyBufferToImageInfo2 copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
            .srcBuffer = vulkan_src_buffer,
            .dstImage = vulkan_dst_image,
            .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .regionCount = 1,
            .pRegions = &region,
        };
        vkCmdCopyBufferToImage2(cmd_buff_, &copy_info);

        gpu_system_->DeallocUploadMemBlock(std::move(upload_mem_block));
    }

    std::future<void> VulkanCommandList::ReadBackAsync(const GpuBuffer& src, const std::function<void(const void* dst_data)>& copy_func)
    {
        switch (src.Heap())
        {
        case GpuHeap::Upload:
        case GpuHeap::ReadBack:
            copy_func(src.Map());
            src.Unmap();
            return {};

        case GpuHeap::Default:
        {
            auto read_back_mem_block = gpu_system_->AllocReadBackMemBlock(src.Size(), gpu_system_->StructuredDataAlignment());

            const auto& vulkan_src = VulkanImp(src);
            vulkan_src.Transition(*this, GpuResourceState::CopySrc);

            const VkBufferCopy2 copy_region{
                .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                .srcOffset = 0,
                .dstOffset = read_back_mem_block.Offset(),
                .size = src.Size(),
            };
            const VkCopyBufferInfo2 copy_info{
                .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                .srcBuffer = vulkan_src.Buffer(),
                .dstBuffer = VulkanImp(*read_back_mem_block.Buffer()).Buffer(),
                .regionCount = 1,
                .pRegions = &copy_region,
            };
            vkCmdCopyBuffer2(cmd_buff_, &copy_info);

            const uint64_t fence_val = VulkanImp(*gpu_system_).ExecuteAndReset(*this, GpuSystem::MaxFenceValue);

            return std::async(
                std::launch::deferred, [this, fence_val, read_back_mem_block = std::move(read_back_mem_block), copy_func]() mutable {
                    gpu_system_->CpuWait(fence_val);

                    const void* buff_data = read_back_mem_block.CpuSpan<std::byte>().data();
                    copy_func(buff_data);

                    gpu_system_->DeallocReadBackMemBlock(std::move(read_back_mem_block));
                });
        }

        default:
            Unreachable("Invalid heap type");
        }
    }

    std::future<void> VulkanCommandList::ReadBackAsync(const GpuTexture& src, uint32_t sub_resource,
        const std::function<void(const void* src_data, uint32_t row_pitch, uint32_t slice_pitch)>& copy_func)
    {
        auto& vulkan_system = VulkanImp(*gpu_system_);
        const auto& vulkan_src = VulkanImp(src);

        uint32_t mip;
        uint32_t array_slice;
        uint32_t plane_slice;
        DecomposeSubResource(sub_resource, src.MipLevels(), src.ArraySize(), mip, array_slice, plane_slice);
        const uint32_t width = src.Width(mip);
        const uint32_t height = src.Height(mip);
        const uint32_t depth = src.Depth(mip);

        const VkDevice vulkan_device = vulkan_system.Device();
        const VkImage vulkan_src_image = vulkan_src.Image();

        VkMemoryRequirements requirements;
        vkGetImageMemoryRequirements(vulkan_device, vulkan_src_image, &requirements);

        auto read_back_mem_block =
            gpu_system_->AllocReadBackMemBlock(static_cast<uint32_t>(requirements.size), gpu_system_->TextureDataAlignment());

        VkBuffer vulkan_dst_buff = VulkanImp(*read_back_mem_block.Buffer()).Buffer();

        vulkan_src.Transition(*this, GpuResourceState::CopySrc);

        const VkBufferImageCopy2 region{
            .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .bufferOffset = read_back_mem_block.Offset(),
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = mip,
                .baseArrayLayer = array_slice,
                .layerCount = 1,
            },
            .imageOffset{
                .x = 0,
                .y = 0,
                .z = 0,
            },
            .imageExtent{
                .width = width,
                .height = height,
                .depth = depth,
            },
        };
        const VkCopyImageToBufferInfo2 copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
            .srcImage = vulkan_src_image,
            .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .dstBuffer = vulkan_dst_buff,
            .regionCount = 1,
            .pRegions = &region,
        };
        vkCmdCopyImageToBuffer2(cmd_buff_, &copy_info);

        const uint64_t fence_val = vulkan_system.ExecuteAndReset(*this, GpuSystem::MaxFenceValue);

        const uint32_t row_pitch = width * FormatSize(src.Format());
        const uint32_t slice_pitch = row_pitch * height;
        return std::async(std::launch::deferred,
            [this, fence_val, read_back_mem_block = std::move(read_back_mem_block), row_pitch, slice_pitch, copy_func]() mutable {
                gpu_system_->CpuWait(fence_val);

                const void* tex_data = read_back_mem_block.CpuSpan<std::byte>().data();
                copy_func(tex_data, row_pitch, slice_pitch);

                gpu_system_->DeallocReadBackMemBlock(std::move(read_back_mem_block));
            });
    }

    void VulkanCommandList::Close()
    {
        TIFVK(vkEndCommandBuffer(cmd_buff_));

        VulkanImp(*cmd_pool_).UnregisterAllocatedCommandBuffer(cmd_buff_);
        closed_ = true;
        cmd_pool_ = nullptr;
    }

    void VulkanCommandList::Reset(GpuCommandPool& cmd_pool)
    {
        cmd_pool_ = &cmd_pool;

        const VkDevice vulkan_device = VulkanImp(*gpu_system_).Device();
        auto& vulkan_cmd_pool = VulkanImp(cmd_pool);

        const VkCommandBufferAllocateInfo command_buff_allocate_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = vulkan_cmd_pool.CmdPool(),
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        TIFVK(vkAllocateCommandBuffers(vulkan_device, &command_buff_allocate_info, &cmd_buff_));

        const VkCommandBufferBeginInfo cmd_buff_begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        };
        TIFVK(vkBeginCommandBuffer(cmd_buff_, &cmd_buff_begin_info));

        vulkan_cmd_pool.RegisterAllocatedCommandBuffer(cmd_buff_);
        closed_ = false;
    }

    void VulkanCommandList::GenWriteDescSet(std::vector<VkWriteDescriptorSet>& write_desc_sets, const VulkanBindingSlots& binding_slots,
        std::string_view shader_name, const GpuCommandList::ShaderBinding& shader_binding, std::span<const VkDescriptorSet> desc_sets)
    {
        if (!binding_slots.cbv_srv_uav.empty())
        {
            for (const auto& [binding, type, name] : binding_slots.cbv_srv_uav)
            {
                if (!name.empty())
                {
                    const char* view_type_name = nullptr;
                    bool found = false;
                    switch (type)
                    {
                    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                        for (const auto& [binding_name, srv] : shader_binding.srvs)
                        {
                            if (binding_name == name)
                            {
                                if (srv != nullptr)
                                {
                                    const auto& vulkan_srv = VulkanImp(*srv);
                                    vulkan_srv.Transition(*this);
                                    write_desc_sets.emplace_back(vulkan_srv.WriteDescSet());
                                }
                                else
                                {
                                    switch (type)
                                    {
                                    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                                        write_desc_sets.emplace_back(NullSampledImageShaderResourceViewWriteDescSet());
                                        break;
                                    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                                        write_desc_sets.emplace_back(NullUniformTexelBufferShaderResourceViewWriteDescSet());
                                        break;
                                    }
                                }
                                found = true;
                                break;
                            }
                        }
                        view_type_name = "SRV";
                        break;

                    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                        for (const auto& [binding_name, uav] : shader_binding.uavs)
                        {
                            if (binding_name == name)
                            {
                                if (uav != nullptr)
                                {
                                    auto& vulkan_uav = VulkanImp(*uav);
                                    vulkan_uav.Transition(*this);
                                    write_desc_sets.emplace_back(vulkan_uav.WriteDescSet());
                                }
                                else
                                {
                                    switch (type)
                                    {
                                    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                                        write_desc_sets.emplace_back(NullStorageImageUnorderedAccessViewWriteDescSet());
                                        break;
                                    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                                        write_desc_sets.emplace_back(NullStorageTexelBufferUnorderedAccessViewWriteDescSet());
                                        break;
                                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                                        write_desc_sets.emplace_back(NullStorageBufferUnorderedAccessViewWriteDescSet());
                                        break;
                                    }
                                }
                                found = true;
                                break;
                            }
                        }
                        view_type_name = "UAV";
                        break;

                    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                        for (const auto& [binding_name, cbv] : shader_binding.cbvs)
                        {
                            if (binding_name == name)
                            {
                                if (cbv != nullptr)
                                {
                                    const auto& vulkan_cbv = VulkanImp(*cbv);
                                    vulkan_cbv.Transition(*this);
                                    write_desc_sets.emplace_back(vulkan_cbv.WriteDescSet());
                                }
                                else
                                {
                                    write_desc_sets.emplace_back(NullUniformBufferConstantBufferViewWriteDescSet());
                                }
                                found = true;
                                break;
                            }
                        }
                        view_type_name = "CBV";
                        break;

                    default:
                        Unreachable("Unsupported descriptor type");
                    }

                    if (found)
                    {
                        auto& write_desc_set = write_desc_sets.back();
                        write_desc_set.dstSet = desc_sets[0];
                        write_desc_set.dstBinding = binding;
                    }
                    else
                    {
                        std::cout << std::format(
                            "{}WARNING: {} {} {} of shader {} is not bound\n", YellowEscape, EndEscape, view_type_name, name, shader_name);
                    }
                }
            }
        }

        if (!binding_slots.samplers.empty())
        {
            for (const auto& [binding, name] : binding_slots.samplers)
            {
                if (!name.empty())
                {
                    bool found = false;
                    for (const auto& [binding_name, sampler] : shader_binding.samplers)
                    {
                        if (binding_name == name)
                        {
                            if (sampler != nullptr)
                            {
                                write_desc_sets.emplace_back(VulkanImp(*sampler).WriteDescSet());
                            }
                            else
                            {
                                write_desc_sets.emplace_back(NullDynamicSamplerWriteDescSet());
                            }
                            found = true;
                            break;
                        }
                    }

                    if (found)
                    {
                        auto& write_desc_set = write_desc_sets.back();
                        write_desc_set.dstSet = desc_sets[1];
                        write_desc_set.dstBinding = binding;
                    }
                    else
                    {
                        std::cout << std::format(
                            "{}WARNING: {} Sampler {} of shader {} is not bound\n", YellowEscape, EndEscape, name, shader_name);
                    }
                }
            }
        }
    }
} // namespace AIHoloImager
