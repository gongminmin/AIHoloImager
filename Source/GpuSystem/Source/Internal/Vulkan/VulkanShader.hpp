// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <span>

#include <volk.h>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuShader.hpp"

#include "../GpuShaderInternal.hpp"
#include "VulkanCommandList.hpp"
#include "VulkanImpDefine.hpp"
#include "VulkanUtil.hpp"

namespace AIHoloImager
{
    struct VulkanBindingSlots
    {
        std::vector<std::tuple<uint32_t, VkDescriptorType, std::string>> cbv_srv_uav;
        std::vector<std::tuple<uint32_t, std::string>> samplers;
    };

    class VulkanRenderPipeline : public GpuRenderPipelineInternal
    {
    public:
        VulkanRenderPipeline(GpuSystem& gpu_system, GpuRenderPipeline::PrimitiveTopology topology, std::span<const ShaderInfo> shaders,
            const GpuVertexAttribs& vertex_attribs, std::span<const GpuStaticSampler> static_samplers,
            const GpuRenderPipeline::States& states);
        ~VulkanRenderPipeline() override;

        VulkanRenderPipeline(VulkanRenderPipeline&& other) noexcept;
        explicit VulkanRenderPipeline(GpuRenderPipelineInternal&& other) noexcept;
        VulkanRenderPipeline& operator=(VulkanRenderPipeline&& other) noexcept;
        GpuRenderPipelineInternal& operator=(GpuRenderPipelineInternal&& other) noexcept override;

        void Bind(GpuCommandList& cmd_list) const override;
        void Bind(VulkanCommandList& cmd_list) const;

        uint32_t NumDescSetLayouts() const noexcept;
        VkDescriptorSetLayout DescSetLayout(uint32_t set) const noexcept;
        VkPipelineLayout PipelineLayout() const noexcept;

        const VulkanBindingSlots& BindingSlots(GpuRenderPipeline::ShaderStage stage) const noexcept;
        const std::string& ShaderName(GpuRenderPipeline::ShaderStage stage) const noexcept;

    private:
        std::vector<VulkanRecyclableObject<VkDescriptorSetLayout>> descriptor_set_layouts_;
        std::vector<std::shared_ptr<VulkanRecyclableObject<VkSampler>>> static_samplers_;
        VulkanRecyclableObject<VkPipelineLayout> pipeline_layout_;
        VulkanRecyclableObject<VkPipeline> pipeline_;
        GpuRenderPipeline::PrimitiveTopology topology_{};

        VulkanBindingSlots binding_slots_[static_cast<size_t>(GpuRenderPipeline::ShaderStage::Num)];
        std::string shader_names_[static_cast<size_t>(GpuRenderPipeline::ShaderStage::Num)];
    };

    VULKAN_DEFINE_IMP(RenderPipeline)

    class VulkanComputePipeline : public GpuComputePipelineInternal
    {
    public:
        VulkanComputePipeline(GpuSystem& gpu_system, const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers);
        ~VulkanComputePipeline() override;

        VulkanComputePipeline(VulkanComputePipeline&& other) noexcept;
        explicit VulkanComputePipeline(GpuComputePipelineInternal&& other) noexcept;
        VulkanComputePipeline& operator=(VulkanComputePipeline&& other) noexcept;
        GpuComputePipelineInternal& operator=(GpuComputePipelineInternal&& other) noexcept override;

        void Bind(GpuCommandList& cmd_list) const override;
        void Bind(VulkanCommandList& cmd_list) const;

        uint32_t NumDescSetLayouts() const noexcept;
        VkDescriptorSetLayout DescSetLayout(uint32_t set) const noexcept;
        VkPipelineLayout PipelineLayout() const noexcept;

        const VulkanBindingSlots& BindingSlots() const noexcept;
        const std::string& ShaderName() const noexcept;

    private:
        std::vector<VulkanRecyclableObject<VkDescriptorSetLayout>> descriptor_set_layouts_;
        std::vector<std::shared_ptr<VulkanRecyclableObject<VkSampler>>> static_samplers_;
        VulkanRecyclableObject<VkPipelineLayout> pipeline_layout_;
        VulkanRecyclableObject<VkPipeline> pipeline_;

        VulkanBindingSlots binding_slots_;
        std::string shader_name_;
    };

    VULKAN_DEFINE_IMP(ComputePipeline)
} // namespace AIHoloImager
