// Copyright (c) 2025 Minmin Gong
//

#include "VulkanShader.hpp"

#include <format>

#include <spirv_reflect.h>

#include "Base/ErrorHandling.hpp"

#include "VulkanConversion.hpp"
#include "VulkanErrorhandling.hpp"
#include "VulkanSampler.hpp"
#include "VulkanSystem.hpp"
#include "VulkanVertexAttrib.hpp"

namespace AIHoloImager
{
    VULKAN_IMP_IMP(RenderPipeline)

    VulkanRenderPipeline::VulkanRenderPipeline(GpuSystem& gpu_system, GpuRenderPipeline::PrimitiveTopology topology,
        std::span<const ShaderInfo> shaders, const GpuVertexAttribs& vertex_attribs, std::span<const GpuStaticSampler> static_samplers,
        const GpuRenderPipeline::States& states)
        : pipeline_layout_(VulkanImp(gpu_system), VK_NULL_HANDLE), pipeline_(VulkanImp(gpu_system), VK_NULL_HANDLE), topology_(topology)
    {
        auto& vulkan_system = VulkanImp(gpu_system);
        const VkDevice vulkan_device = vulkan_system.Device();

        std::vector<std::vector<VkDescriptorSetLayoutBinding>> desc_layout_bindings;
        std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
        std::vector<VkShaderModule> shader_modules;
        for (size_t s = 0; s < shaders.size(); ++s)
        {
            shader_names_[s] = shaders[s].name;

            const auto bytecode = shaders[s].bytecodes[static_cast<uint32_t>(ShaderInfo::BytecodeFormat::Spv)];
            if (bytecode.empty())
            {
                continue;
            }

            SpvReflectShaderModule reflect_module{};
            TIFSPVRFL(spvReflectCreateShaderModule(bytecode.size(), bytecode.data(), &reflect_module));

            uint32_t count = 0;
            TIFSPVRFL(spvReflectEnumerateDescriptorSets(&reflect_module, &count, nullptr));
            auto reflect_desc_sets = std::make_unique<SpvReflectDescriptorSet*[]>(count);
            TIFSPVRFL(spvReflectEnumerateDescriptorSets(&reflect_module, &count, reflect_desc_sets.get()));

            desc_layout_bindings.resize(std::max(desc_layout_bindings.size(), static_cast<size_t>(count)));

            VkShaderStageFlags stage_flags;
            VkShaderStageFlagBits stage_flag_bit;
            switch (static_cast<GpuRenderPipeline::ShaderStage>(s))
            {
            case GpuRenderPipeline::ShaderStage::Vertex:
                stage_flags = stage_flag_bit = VK_SHADER_STAGE_VERTEX_BIT;
                break;
            case GpuRenderPipeline::ShaderStage::Pixel:
                stage_flags = stage_flag_bit = VK_SHADER_STAGE_FRAGMENT_BIT;
                break;
            case GpuRenderPipeline::ShaderStage::Geometry:
                stage_flags = stage_flag_bit = VK_SHADER_STAGE_GEOMETRY_BIT;
                break;

            default:
                Unreachable("Invalid shader stage");
            }

            for (uint32_t set = 0; set < count; ++set)
            {
                uint32_t static_sampler_index = 0;
                for (uint32_t i = 0; i < reflect_desc_sets[set]->binding_count; ++i)
                {
                    const auto& reflect_binding = *reflect_desc_sets[set]->bindings[i];

                    auto& desc_layout_binding = desc_layout_bindings[set].emplace_back(VkDescriptorSetLayoutBinding{
                        .binding = reflect_binding.binding,
                        .descriptorCount = reflect_binding.count,
                        .stageFlags = stage_flags,
                    });

                    if (reflect_binding.descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER)
                    {
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                        if (reflect_binding.set == 0)
                        {
                            const auto& vulkan_sampler = VulkanImp(static_samplers[static_sampler_index]);
                            auto& new_sampler = static_samplers_.emplace_back(vulkan_sampler.Sampler());
                            desc_layout_binding.pImmutableSamplers = &new_sampler->Object();
                            ++static_sampler_index;
                        }
                        else
                        {
                            binding_slots_[s].samplers.emplace_back(reflect_binding.binding, reflect_binding.name);
                        }
                    }
                    else
                    {
                        switch (reflect_binding.descriptor_type)
                        {
                        case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                            break;
                        case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                            break;
                        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                            break;
                        case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                            break;
                        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                            break;
                        case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                            break;
                        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                            break;
                        case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                            break;
                        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
                            break;
                        case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
                            break;
                        case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
                            desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                            break;

                        default:
                            Unreachable("Unsupported descriptor type");
                        }

                        binding_slots_[s].cbv_srv_uav.emplace_back(
                            reflect_binding.binding, desc_layout_binding.descriptorType, reflect_binding.name);
                    }
                }
            }

            spvReflectDestroyShaderModule(&reflect_module);

            VkShaderModule& shader_module = shader_modules.emplace_back();
            const VkShaderModuleCreateInfo module_create_info{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = bytecode.size(),
                .pCode = reinterpret_cast<const uint32_t*>(bytecode.data()),
            };
            TIFVK(vkCreateShaderModule(vulkan_device, &module_create_info, nullptr, &shader_module));

            shader_stages.push_back(VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = stage_flag_bit,
                .module = shader_module,
                .pName = "main",
            });
        }

        descriptor_set_layouts_.resize(desc_layout_bindings.size());
        auto descriptor_set_layout_ptrs = std::make_unique<VkDescriptorSetLayout[]>(desc_layout_bindings.size());
        for (size_t set = 0; set < desc_layout_bindings.size(); ++set)
        {
            descriptor_set_layouts_[set] = VulkanRecyclableObject<VkDescriptorSetLayout>(vulkan_system, VK_NULL_HANDLE);

            const VkDescriptorSetLayoutCreateInfo desc_layout_create_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = static_cast<uint32_t>(desc_layout_bindings[set].size()),
                .pBindings = desc_layout_bindings[set].data(),
            };
            TIFVK(vkCreateDescriptorSetLayout(vulkan_device, &desc_layout_create_info, nullptr, &descriptor_set_layouts_[set].Object()));
            descriptor_set_layout_ptrs[set] = descriptor_set_layouts_[set].Object();
        }

        const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = static_cast<uint32_t>(desc_layout_bindings.size()),
            .pSetLayouts = descriptor_set_layout_ptrs.get(),
        };
        TIFVK(vkCreatePipelineLayout(vulkan_device, &pipeline_layout_create_info, nullptr, &pipeline_layout_.Object()));

        const auto& vulkan_vertex_attribs = VulkanImp(vertex_attribs);
        const auto input_binding = vulkan_vertex_attribs.InputBindings();
        const auto input_attribs = vulkan_vertex_attribs.InputAttribs();
        const VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = static_cast<uint32_t>(input_binding.size()),
            .pVertexBindingDescriptions = input_binding.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(input_attribs.size()),
            .pVertexAttributeDescriptions = input_attribs.data(),
        };

        VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .primitiveRestartEnable = (topology_ == GpuRenderPipeline::PrimitiveTopology::TriangleStrip),
        };
        switch (topology_)
        {
        case GpuRenderPipeline::PrimitiveTopology::PointList:
            input_assembly_state_create_info.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            break;
        case GpuRenderPipeline::PrimitiveTopology::TriangleList:
            input_assembly_state_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            break;
        case GpuRenderPipeline::PrimitiveTopology::TriangleStrip:
            input_assembly_state_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
            break;

        default:
            Unreachable("Invalid topology");
        }

        const VkPipelineViewportStateCreateInfo viewport_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };

        VkPipelineRasterizationStateCreateInfo rasterization_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .lineWidth = 1.0f,
        };
        switch (states.cull_mode)
        {
        case GpuRenderPipeline::CullMode::None:
            rasterization_state_create_info.cullMode = VK_CULL_MODE_NONE;
            break;
        case GpuRenderPipeline::CullMode::ClockWise:
            rasterization_state_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
            break;
        case GpuRenderPipeline::CullMode::CounterClockWise:
            rasterization_state_create_info.cullMode = VK_CULL_MODE_FRONT_BIT;
            break;
        }

        VkPipelineRasterizationConservativeStateCreateInfoEXT conservative_raster_state_create_info;
        if (states.conservative_raster)
        {
            VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservative_raster_props{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT,
            };
            VkPhysicalDeviceProperties2KHR device_props{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR,
                .pNext = &conservative_raster_props,
            };
            vkGetPhysicalDeviceProperties2(vulkan_system.PhysicalDevice(), &device_props);

            conservative_raster_state_create_info = VkPipelineRasterizationConservativeStateCreateInfoEXT{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
                .conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT,
                .extraPrimitiveOverestimationSize = conservative_raster_props.maxExtraPrimitiveOverestimationSize,
            };

            rasterization_state_create_info.pNext = &conservative_raster_state_create_info;
        }

        const VkPipelineMultisampleStateCreateInfo multisample_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .pSampleMask = nullptr,
        };

        const VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = states.depth_enable,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS,
        };

        auto blend_attachment_states = std::make_unique<VkPipelineColorBlendAttachmentState[]>(states.rtv_formats.size());
        for (size_t i = 0; i < states.rtv_formats.size(); ++i)
        {
            blend_attachment_states[i] = {
                .blendEnable = VK_FALSE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            };
        }

        const VkPipelineColorBlendStateCreateInfo color_blend_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_NO_OP,
            .attachmentCount = static_cast<uint32_t>(states.rtv_formats.size()),
            .pAttachments = blend_attachment_states.get(),
        };

        const VkDynamicState dynamic_state_enables[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        const VkPipelineDynamicStateCreateInfo dynamic_state_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(std::size(dynamic_state_enables)),
            .pDynamicStates = dynamic_state_enables,
        };

        auto rtv_formats = std::make_unique<VkFormat[]>(states.rtv_formats.size());
        for (size_t i = 0; i < states.rtv_formats.size(); ++i)
        {
            rtv_formats[i] = ToVkFormat(states.rtv_formats[i]);
        }
        const VkFormat depth_format = ToVkFormat(states.dsv_format);
        VkFormat stencil_format = VK_FORMAT_UNDEFINED;
        if (IsStencilFormat(states.dsv_format))
        {
            stencil_format = depth_format;
        }

        const VkPipelineRenderingCreateInfo pipeline_rendering_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = static_cast<uint32_t>(states.rtv_formats.size()),
            .pColorAttachmentFormats = rtv_formats.get(),
            .depthAttachmentFormat = depth_format,
            .stencilAttachmentFormat = stencil_format,
        };

        const VkGraphicsPipelineCreateInfo pipeline_create_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &pipeline_rendering_create_info,
            .stageCount = static_cast<uint32_t>(shader_stages.size()),
            .pStages = shader_stages.data(),
            .pVertexInputState = &vertex_input_state_create_info,
            .pInputAssemblyState = &input_assembly_state_create_info,
            .pViewportState = &viewport_state_create_info,
            .pRasterizationState = &rasterization_state_create_info,
            .pMultisampleState = &multisample_state_create_info,
            .pDepthStencilState = &depth_stencil_state_create_info,
            .pColorBlendState = &color_blend_state_create_info,
            .pDynamicState = &dynamic_state_create_info,
            .layout = pipeline_layout_.Object(),
        };
        TIFVK(vkCreateGraphicsPipelines(vulkan_device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline_.Object()));

        for (auto& shader_module : shader_modules)
        {
            vkDestroyShaderModule(vulkan_device, shader_module, nullptr);
        }
    }

    VulkanRenderPipeline::~VulkanRenderPipeline() = default;

    VulkanRenderPipeline::VulkanRenderPipeline(VulkanRenderPipeline&& other) noexcept = default;
    VulkanRenderPipeline::VulkanRenderPipeline(GpuRenderPipelineInternal&& other) noexcept
        : VulkanRenderPipeline(static_cast<VulkanRenderPipeline&&>(other))
    {
    }

    VulkanRenderPipeline& VulkanRenderPipeline::operator=(VulkanRenderPipeline&& other) noexcept = default;
    GpuRenderPipelineInternal& VulkanRenderPipeline::operator=(GpuRenderPipelineInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanRenderPipeline&&>(other));
    }

    void VulkanRenderPipeline::Bind(GpuCommandList& cmd_list) const
    {
        this->Bind(VulkanImp(cmd_list));
    }

    void VulkanRenderPipeline::Bind(VulkanCommandList& cmd_list) const
    {
        VkCommandBuffer cmd_buff = cmd_list.CommandBuffer();
        vkCmdBindPipeline(cmd_buff, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.Object());
    }

    uint32_t VulkanRenderPipeline::NumDescSetLayouts() const noexcept
    {
        return static_cast<uint32_t>(descriptor_set_layouts_.size());
    }
    VkDescriptorSetLayout VulkanRenderPipeline::DescSetLayout(uint32_t set) const noexcept
    {
        return descriptor_set_layouts_[set].Object();
    }

    VkPipelineLayout VulkanRenderPipeline::PipelineLayout() const noexcept
    {
        return pipeline_layout_.Object();
    }

    const VulkanBindingSlots& VulkanRenderPipeline::BindingSlots(GpuRenderPipeline::ShaderStage stage) const noexcept
    {
        return binding_slots_[static_cast<uint32_t>(stage)];
    }

    const std::string& VulkanRenderPipeline::ShaderName(GpuRenderPipeline::ShaderStage stage) const noexcept
    {
        return shader_names_[static_cast<uint32_t>(stage)];
    }


    VULKAN_IMP_IMP(ComputePipeline)

    VulkanComputePipeline::VulkanComputePipeline(
        GpuSystem& gpu_system, const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers)
        : shader_name_(shader.name), pipeline_layout_(VulkanImp(gpu_system), VK_NULL_HANDLE),
          pipeline_(VulkanImp(gpu_system), VK_NULL_HANDLE)
    {
        const VkDevice vulkan_device = VulkanImp(gpu_system).Device();

        const auto bytecode = shader.bytecodes[static_cast<uint32_t>(ShaderInfo::BytecodeFormat::Spv)];

        SpvReflectShaderModule reflect_module{};
        TIFSPVRFL(spvReflectCreateShaderModule(bytecode.size(), bytecode.data(), &reflect_module));

        uint32_t count = 0;
        TIFSPVRFL(spvReflectEnumerateDescriptorSets(&reflect_module, &count, nullptr));
        auto reflect_desc_sets = std::make_unique<SpvReflectDescriptorSet*[]>(count);
        TIFSPVRFL(spvReflectEnumerateDescriptorSets(&reflect_module, &count, reflect_desc_sets.get()));

        descriptor_set_layouts_.resize(count);
        std::vector<VkDescriptorSetLayout> descriptor_set_layout_ptrs(count);

        for (uint32_t set = 0; set < count; ++set)
        {
            descriptor_set_layouts_[set] = VulkanRecyclableObject<VkDescriptorSetLayout>{VulkanImp(gpu_system), VK_NULL_HANDLE};

            std::vector<VkDescriptorSetLayoutBinding> desc_layout_bindings;
            uint32_t static_sampler_index = 0;
            for (uint32_t i = 0; i < reflect_desc_sets[set]->binding_count; ++i)
            {
                const auto& reflect_binding = *reflect_desc_sets[set]->bindings[i];

                auto& desc_layout_binding = desc_layout_bindings.emplace_back(VkDescriptorSetLayoutBinding{
                    .binding = reflect_binding.binding,
                    .descriptorCount = reflect_binding.count,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                });

                if (reflect_binding.descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER)
                {
                    desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                    if (reflect_binding.set == 0)
                    {
                        const auto& vulkan_sampler = VulkanImp(static_samplers[static_sampler_index]);
                        auto& new_sampler = static_samplers_.emplace_back(vulkan_sampler.Sampler());
                        desc_layout_binding.pImmutableSamplers = &new_sampler->Object();
                        ++static_sampler_index;
                    }
                    else
                    {
                        binding_slots_.samplers.emplace_back(reflect_binding.binding, reflect_binding.name);
                    }
                }
                else
                {
                    switch (reflect_binding.descriptor_type)
                    {
                    case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
                        break;
                    case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
                        desc_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                        break;

                    default:
                        Unreachable("Unsupported descriptor type");
                    }

                    binding_slots_.cbv_srv_uav.emplace_back(
                        reflect_binding.binding, desc_layout_binding.descriptorType, reflect_binding.name);
                }
            }

            const VkDescriptorSetLayoutCreateInfo desc_layout_create_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = static_cast<uint32_t>(desc_layout_bindings.size()),
                .pBindings = desc_layout_bindings.data(),
            };
            TIFVK(vkCreateDescriptorSetLayout(vulkan_device, &desc_layout_create_info, nullptr, &descriptor_set_layouts_[set].Object()));
            descriptor_set_layout_ptrs[set] = descriptor_set_layouts_[set].Object();
        }

        spvReflectDestroyShaderModule(&reflect_module);

        const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = static_cast<uint32_t>(descriptor_set_layout_ptrs.size()),
            .pSetLayouts = descriptor_set_layout_ptrs.data(),
        };
        TIFVK(vkCreatePipelineLayout(vulkan_device, &pipeline_layout_create_info, nullptr, &pipeline_layout_.Object()));

        VkShaderModule shader_module;
        const VkShaderModuleCreateInfo module_create_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = bytecode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(bytecode.data()),
        };
        TIFVK(vkCreateShaderModule(vulkan_device, &module_create_info, nullptr, &shader_module));

        const VkComputePipelineCreateInfo compute_pipeline_create_info{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .flags = 0,
            .stage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
            },
            .layout = pipeline_layout_.Object(),
        };
        TIFVK(vkCreateComputePipelines(vulkan_device, VK_NULL_HANDLE, 1, &compute_pipeline_create_info, nullptr, &pipeline_.Object()));

        vkDestroyShaderModule(vulkan_device, shader_module, nullptr);
    }

    VulkanComputePipeline::~VulkanComputePipeline() = default;

    VulkanComputePipeline::VulkanComputePipeline(VulkanComputePipeline&& other) noexcept = default;
    VulkanComputePipeline::VulkanComputePipeline(GpuComputePipelineInternal&& other) noexcept
        : VulkanComputePipeline(static_cast<VulkanComputePipeline&&>(other))
    {
    }

    VulkanComputePipeline& VulkanComputePipeline::operator=(VulkanComputePipeline&& other) noexcept = default;
    GpuComputePipelineInternal& VulkanComputePipeline::operator=(GpuComputePipelineInternal&& other) noexcept
    {
        return this->operator=(static_cast<VulkanComputePipeline&&>(other));
    }

    void VulkanComputePipeline::Bind(GpuCommandList& cmd_list) const
    {
        this->Bind(VulkanImp(cmd_list));
    }

    void VulkanComputePipeline::Bind(VulkanCommandList& cmd_list) const
    {
        VkCommandBuffer cmd_buff = cmd_list.CommandBuffer();
        vkCmdBindPipeline(cmd_buff, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.Object());
    }

    uint32_t VulkanComputePipeline::NumDescSetLayouts() const noexcept
    {
        return static_cast<uint32_t>(descriptor_set_layouts_.size());
    }
    VkDescriptorSetLayout VulkanComputePipeline::DescSetLayout(uint32_t set) const noexcept
    {
        return descriptor_set_layouts_[set].Object();
    }

    VkPipelineLayout VulkanComputePipeline::PipelineLayout() const noexcept
    {
        return pipeline_layout_.Object();
    }

    const VulkanBindingSlots& VulkanComputePipeline::BindingSlots() const noexcept
    {
        return binding_slots_;
    }

    const std::string& VulkanComputePipeline::ShaderName() const noexcept
    {
        return shader_name_;
    }
} // namespace AIHoloImager
