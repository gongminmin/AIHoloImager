// Copyright (c) 2025 Minmin Gong
//

#include "D3D12Shader.hpp"

#include <format>

#include "Base/ErrorHandling.hpp"
#include "Base/MiniWindows.hpp"

#include "D3D12Conversion.hpp"
#include "D3D12Sampler.hpp"
#include "D3D12System.hpp"
#include "D3D12VertexLayout.hpp"

namespace
{
    D3D12_ROOT_PARAMETER CreateRootParameterAsDescriptorTable(const D3D12_DESCRIPTOR_RANGE* descriptor_ranges,
        uint32_t num_descriptor_ranges, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        D3D12_ROOT_PARAMETER ret{
            .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            .DescriptorTable{
                .NumDescriptorRanges = num_descriptor_ranges,
                .pDescriptorRanges = descriptor_ranges,
            },
            .ShaderVisibility = visibility,
        };
        return ret;
    }

    D3D12_ROOT_PARAMETER CreateRootParameterAsConstantBufferView(
        uint32_t shader_register, uint32_t register_space = 0, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        D3D12_ROOT_PARAMETER ret{
            .ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV,
            .Descriptor{
                .ShaderRegister = shader_register,
                .RegisterSpace = register_space,
            },
            .ShaderVisibility = visibility,
        };
        return ret;
    }

    struct ShaderReflectionInfo
    {
        uint32_t num_cbvs = 0;
        uint32_t num_srvs = 0;
        uint32_t num_uavs = 0;
        uint32_t num_dynamic_samplers = 0;
    };
} // namespace

namespace AIHoloImager
{
    D3D12_IMP_IMP(RenderPipeline)

    D3D12RenderPipeline::D3D12RenderPipeline(GpuSystem& gpu_system, GpuRenderPipeline::PrimitiveTopology topology,
        std::span<const ShaderInfo> shaders, const GpuVertexLayout& vertex_layout, std::span<const GpuStaticSampler> static_samplers,
        const GpuRenderPipeline::States& states)
        : root_sig_(D3D12Imp(gpu_system), nullptr), pso_(D3D12Imp(gpu_system), nullptr), topology_(topology)
    {
        ShaderReflectionInfo shader_rfls[static_cast<size_t>(GpuRenderPipeline::ShaderStage::Num)] = {};
        uint32_t num_desc_ranges = 0;
        for (size_t s = 0; s < shaders.size(); ++s)
        {
            shader_names_[s] = shaders[s].name;

            const auto bytecode = shaders[s].bytecodes[static_cast<uint32_t>(ShaderInfo::BytecodeFormat::Dxil)];
            if (bytecode.empty())
            {
                continue;
            }

            ComPtr<ID3D12ShaderReflection> reflection = D3D12Imp(gpu_system).ShaderReflect(bytecode);

            D3D12_SHADER_DESC shader_desc;
            TIFHR(reflection->GetDesc(&shader_desc));

            auto& binding_slots = binding_slots_[s];

            auto& shader_rfl = shader_rfls[s];
            for (uint32_t resource_index = 0; resource_index < shader_desc.BoundResources; ++resource_index)
            {
                D3D12_SHADER_INPUT_BIND_DESC bind_desc;
                reflection->GetResourceBindingDesc(resource_index, &bind_desc);

                switch (bind_desc.Type)
                {
                case D3D_SIT_CBUFFER:
                case D3D_SIT_TBUFFER:
                    shader_rfl.num_cbvs = std::max(shader_rfl.num_cbvs, bind_desc.BindPoint + 1);
                    binding_slots.cbvs.resize(shader_rfl.num_cbvs);
                    binding_slots.cbvs[bind_desc.BindPoint] = bind_desc.Name;
                    break;

                case D3D_SIT_TEXTURE:
                case D3D_SIT_STRUCTURED:
                case D3D_SIT_BYTEADDRESS:
                    shader_rfl.num_srvs = std::max(shader_rfl.num_srvs, bind_desc.BindPoint + 1);
                    binding_slots.srvs.resize(shader_rfl.num_srvs);
                    binding_slots.srvs[bind_desc.BindPoint] = bind_desc.Name;
                    break;

                case D3D_SIT_UAV_RWTYPED:
                case D3D_SIT_UAV_RWSTRUCTURED:
                case D3D_SIT_UAV_RWBYTEADDRESS:
                case D3D_SIT_UAV_APPEND_STRUCTURED:
                case D3D_SIT_UAV_CONSUME_STRUCTURED:
                case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER:
                    shader_rfl.num_uavs = std::max(shader_rfl.num_uavs, bind_desc.BindPoint + 1);
                    binding_slots.uavs.resize(shader_rfl.num_uavs);
                    binding_slots.uavs[bind_desc.BindPoint] = bind_desc.Name;
                    break;

                case D3D_SIT_SAMPLER:
                    if (bind_desc.Space != 0)
                    {
                        shader_rfl.num_dynamic_samplers = std::max(shader_rfl.num_dynamic_samplers, bind_desc.BindPoint + 1);
                        binding_slots.samplers.resize(shader_rfl.num_dynamic_samplers);
                        binding_slots.samplers[bind_desc.BindPoint] = bind_desc.Name;
                    }
                    break;

                default:
                    Unreachable("Unknown bind type.");
                }
            }

            num_desc_ranges += (shader_rfl.num_srvs ? 1 : 0) + (shader_rfl.num_uavs ? 1 : 0) + (shader_rfl.num_dynamic_samplers ? 1 : 0);
        }

        auto ranges = std::make_unique<D3D12_DESCRIPTOR_RANGE[]>(num_desc_ranges);
        uint32_t range_index = 0;
        uint32_t num_root_params = num_desc_ranges;
        for (const auto& shader_rfl : shader_rfls)
        {
            if (shader_rfl.num_srvs != 0)
            {
                ranges[range_index] = {
                    .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    .NumDescriptors = shader_rfl.num_srvs,
                    .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                };
                ++range_index;
            }
            if (shader_rfl.num_uavs != 0)
            {
                ranges[range_index] = {
                    .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                    .NumDescriptors = shader_rfl.num_uavs,
                    .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                };
                ++range_index;
            }
            if (shader_rfl.num_dynamic_samplers != 0)
            {
                ranges[range_index] = {
                    .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
                    .NumDescriptors = shader_rfl.num_dynamic_samplers,
                    .RegisterSpace = 1,
                    .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                };
                ++range_index;
            }

            num_root_params += shader_rfl.num_cbvs;
        }

        std::unique_ptr<D3D12_ROOT_PARAMETER[]> root_params;
        if (num_root_params > 0)
        {
            root_params = std::make_unique<D3D12_ROOT_PARAMETER[]>(num_root_params);
            uint32_t root_index = 0;
            range_index = 0;
            for (size_t s = 0; s < shaders.size(); ++s)
            {
                const auto& shader_rfl = shader_rfls[s];

                D3D12_SHADER_VISIBILITY visibility;
                switch (static_cast<GpuRenderPipeline::ShaderStage>(s))
                {
                case GpuRenderPipeline::ShaderStage::Vertex:
                    visibility = D3D12_SHADER_VISIBILITY_VERTEX;
                    break;

                case GpuRenderPipeline::ShaderStage::Pixel:
                    visibility = D3D12_SHADER_VISIBILITY_PIXEL;
                    break;

                case GpuRenderPipeline::ShaderStage::Geometry:
                    visibility = D3D12_SHADER_VISIBILITY_GEOMETRY;
                    break;

                default:
                    visibility = D3D12_SHADER_VISIBILITY_ALL;
                    break;
                }

                if (shader_rfl.num_srvs != 0)
                {
                    root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[range_index], 1, visibility);
                    ++root_index;
                    ++range_index;
                }
                if (shader_rfl.num_uavs != 0)
                {
                    root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[range_index], 1, visibility);
                    ++root_index;
                    ++range_index;
                }
                if (shader_rfl.num_dynamic_samplers != 0)
                {
                    root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[range_index], 1, visibility);
                    ++root_index;
                    ++range_index;
                }
                for (uint32_t i = 0; i < shader_rfl.num_cbvs; ++i)
                {
                    root_params[root_index] = CreateRootParameterAsConstantBufferView(i, 0, visibility);
                    ++root_index;
                }
            }
        }

        auto& d3d12_vertex_layout = D3D12Imp(vertex_layout);

        const auto vb_slot_strides = d3d12_vertex_layout.SlotStrides();
        vb_slot_strides_ = std::vector(vb_slot_strides.begin(), vb_slot_strides.end());

        const auto input_elems = d3d12_vertex_layout.InputElementDescs();
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        if (!input_elems.empty())
        {
            flags |= D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        }

        auto d3d12_static_samplers = std::make_unique<D3D12_STATIC_SAMPLER_DESC[]>(static_samplers.size());
        for (uint32_t i = 0; i < static_samplers.size(); ++i)
        {
            d3d12_static_samplers[i] = D3D12Imp(static_samplers[i]).SamplerDesc(i);
        }

        const D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {
            .NumParameters = num_root_params,
            .pParameters = root_params.get(),
            .NumStaticSamplers = static_cast<uint32_t>(static_samplers.size()),
            .pStaticSamplers = d3d12_static_samplers.get(),
            .Flags = flags,
        };

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        HRESULT hr = ::D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.Put(), error.Put());
        if (FAILED(hr))
        {
            ::OutputDebugStringA(
                std::format("D3D12SerializeRootSignature failed: {}\n", static_cast<const char*>(error->GetBufferPointer())).c_str());
            TIFHR(hr);
        }

        ID3D12Device* d3d12_device = D3D12Imp(gpu_system).Device();

        TIFHR(d3d12_device->CreateRootSignature(
            1, blob->GetBufferPointer(), blob->GetBufferSize(), UuidOf<ID3D12RootSignature>(), root_sig_.Object().PutVoid()));

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root_sig_.Object().Get();
        for (size_t s = 0; s < shaders.size(); ++s)
        {
            const auto bytecode = shaders[s].bytecodes[static_cast<uint32_t>(ShaderInfo::BytecodeFormat::Dxil)];
            const D3D12_SHADER_BYTECODE shader_bytecode{
                .pShaderBytecode = bytecode.data(),
                .BytecodeLength = bytecode.size(),
            };

            switch (static_cast<GpuRenderPipeline::ShaderStage>(s))
            {
            case GpuRenderPipeline::ShaderStage::Vertex:
                pso_desc.VS = shader_bytecode;
                break;

            case GpuRenderPipeline::ShaderStage::Pixel:
                pso_desc.PS = shader_bytecode;
                break;

            case GpuRenderPipeline::ShaderStage::Geometry:
                pso_desc.GS = shader_bytecode;
                break;

            default:
                throw std::runtime_error("Not supported yet");
            }
        }
        for (uint32_t i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
        {
            pso_desc.BlendState.RenderTarget[i] = {
                .SrcBlend = D3D12_BLEND_ONE,
                .DestBlend = D3D12_BLEND_ZERO,
                .BlendOp = D3D12_BLEND_OP_ADD,
                .SrcBlendAlpha = D3D12_BLEND_ONE,
                .DestBlendAlpha = D3D12_BLEND_ZERO,
                .BlendOpAlpha = D3D12_BLEND_OP_ADD,
                .LogicOp = D3D12_LOGIC_OP_NOOP,
                .RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL,
            };
        }
        pso_desc.SampleMask = UINT_MAX;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        switch (states.cull_mode)
        {
        case GpuRenderPipeline::CullMode::None:
            pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            break;
        case GpuRenderPipeline::CullMode::ClockWise:
            pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
            break;
        case GpuRenderPipeline::CullMode::CounterClockWise:
            pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_FRONT;
            break;
        }
        pso_desc.RasterizerState.FrontCounterClockwise = TRUE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.RasterizerState.ConservativeRaster =
            states.conservative_raster ? D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON : D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
        pso_desc.DepthStencilState.DepthEnable = states.depth_enable;
        pso_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        pso_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        pso_desc.InputLayout.pInputElementDescs = input_elems.data();
        pso_desc.InputLayout.NumElements = static_cast<uint32_t>(input_elems.size());
        pso_desc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
        switch (topology)
        {
        case GpuRenderPipeline::PrimitiveTopology::PointList:
            pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
            break;
        case GpuRenderPipeline::PrimitiveTopology::TriangleList:
            pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            break;
        case GpuRenderPipeline::PrimitiveTopology::TriangleStrip:
        default:
            pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            break;
        }
        pso_desc.NumRenderTargets = static_cast<uint32_t>(states.rtv_formats.size());
        for (size_t i = 0; i < states.rtv_formats.size(); ++i)
        {
            pso_desc.RTVFormats[i] = ToDxgiFormat(states.rtv_formats[i]);
        }
        pso_desc.DSVFormat = ToDxgiFormat(states.dsv_format);
        pso_desc.SampleDesc = {
            .Count = 1,
            .Quality = 0,
        };
        pso_desc.NodeMask = 0;
        pso_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        TIFHR(d3d12_device->CreateGraphicsPipelineState(&pso_desc, UuidOf<ID3D12PipelineState>(), pso_.Object().PutVoid()));
    }

    D3D12RenderPipeline::~D3D12RenderPipeline() = default;

    D3D12RenderPipeline::D3D12RenderPipeline(D3D12RenderPipeline&& other) noexcept = default;
    D3D12RenderPipeline::D3D12RenderPipeline(GpuRenderPipelineInternal&& other) noexcept
        : D3D12RenderPipeline(static_cast<D3D12RenderPipeline&&>(other))
    {
    }

    D3D12RenderPipeline& D3D12RenderPipeline::operator=(D3D12RenderPipeline&& other) noexcept = default;
    GpuRenderPipelineInternal& D3D12RenderPipeline::operator=(GpuRenderPipelineInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12RenderPipeline&&>(other));
    }

    void D3D12RenderPipeline::Bind(GpuCommandList& cmd_list) const
    {
        this->Bind(D3D12Imp(cmd_list));
    }

    void D3D12RenderPipeline::Bind(D3D12CommandList& cmd_list) const
    {
        auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

        d3d12_cmd_list->IASetPrimitiveTopology(ToD3D12PrimitiveTopology(topology_));

        d3d12_cmd_list->SetPipelineState(pso_.Object().Get());
        d3d12_cmd_list->SetGraphicsRootSignature(root_sig_.Object().Get());
    }

    const D3D12BindingSlots& D3D12RenderPipeline::BindingSlots(GpuRenderPipeline::ShaderStage stage) const noexcept
    {
        return binding_slots_[static_cast<uint32_t>(stage)];
    }

    const std::string& D3D12RenderPipeline::ShaderName(GpuRenderPipeline::ShaderStage stage) const noexcept
    {
        return shader_names_[static_cast<uint32_t>(stage)];
    }

    std::span<const uint32_t> D3D12RenderPipeline::VertexBufferSlotStrides() const noexcept
    {
        return std::span(vb_slot_strides_);
    }


    D3D12_IMP_IMP(ComputePipeline)

    D3D12ComputePipeline::D3D12ComputePipeline(
        GpuSystem& gpu_system, const ShaderInfo& shader, std::span<const GpuStaticSampler> static_samplers)
        : root_sig_(D3D12Imp(gpu_system), nullptr), pso_(D3D12Imp(gpu_system), nullptr), shader_name_(shader.name)
    {
        const auto bytecode = shader.bytecodes[static_cast<uint32_t>(ShaderInfo::BytecodeFormat::Dxil)];

        ShaderReflectionInfo shader_rfl{};
        {
            ComPtr<ID3D12ShaderReflection> reflection = D3D12Imp(gpu_system).ShaderReflect(bytecode);

            D3D12_SHADER_DESC shader_desc;
            TIFHR(reflection->GetDesc(&shader_desc));

            for (uint32_t resource_index = 0; resource_index < shader_desc.BoundResources; ++resource_index)
            {
                D3D12_SHADER_INPUT_BIND_DESC bind_desc;
                reflection->GetResourceBindingDesc(resource_index, &bind_desc);

                switch (bind_desc.Type)
                {
                case D3D_SIT_CBUFFER:
                case D3D_SIT_TBUFFER:
                    shader_rfl.num_cbvs = std::max(shader_rfl.num_cbvs, bind_desc.BindPoint + 1);
                    binding_slots_.cbvs.resize(shader_rfl.num_cbvs);
                    binding_slots_.cbvs[bind_desc.BindPoint] = bind_desc.Name;
                    break;

                case D3D_SIT_TEXTURE:
                case D3D_SIT_STRUCTURED:
                case D3D_SIT_BYTEADDRESS:
                    shader_rfl.num_srvs = std::max(shader_rfl.num_srvs, bind_desc.BindPoint + 1);
                    binding_slots_.srvs.resize(shader_rfl.num_srvs);
                    binding_slots_.srvs[bind_desc.BindPoint] = bind_desc.Name;
                    break;

                case D3D_SIT_UAV_RWTYPED:
                case D3D_SIT_UAV_RWSTRUCTURED:
                case D3D_SIT_UAV_RWBYTEADDRESS:
                case D3D_SIT_UAV_APPEND_STRUCTURED:
                case D3D_SIT_UAV_CONSUME_STRUCTURED:
                case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER:
                    shader_rfl.num_uavs = std::max(shader_rfl.num_uavs, bind_desc.BindPoint + 1);
                    binding_slots_.uavs.resize(shader_rfl.num_uavs);
                    binding_slots_.uavs[bind_desc.BindPoint] = bind_desc.Name;
                    break;

                case D3D_SIT_SAMPLER:
                    if (bind_desc.Space != 0)
                    {
                        shader_rfl.num_dynamic_samplers = std::max(shader_rfl.num_dynamic_samplers, bind_desc.BindPoint + 1);
                        binding_slots_.samplers.resize(shader_rfl.num_dynamic_samplers);
                        binding_slots_.samplers[bind_desc.BindPoint] = bind_desc.Name;
                    }
                    break;

                default:
                    Unreachable("Unknown bind type.");
                }
            }
        }

        const uint32_t num_desc_ranges =
            (shader_rfl.num_srvs ? 1 : 0) + (shader_rfl.num_uavs ? 1 : 0) + (shader_rfl.num_dynamic_samplers ? 1 : 0);
        auto ranges = std::make_unique<D3D12_DESCRIPTOR_RANGE[]>(num_desc_ranges);
        uint32_t range_index = 0;
        if (shader_rfl.num_srvs != 0)
        {
            ranges[range_index] = {
                .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                .NumDescriptors = shader_rfl.num_srvs,
                .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            };
            ++range_index;
        }
        if (shader_rfl.num_uavs != 0)
        {
            ranges[range_index] = {
                .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                .NumDescriptors = shader_rfl.num_uavs,
                .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            };
            ++range_index;
        }
        if (shader_rfl.num_dynamic_samplers != 0)
        {
            ranges[range_index] = {
                .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
                .NumDescriptors = shader_rfl.num_dynamic_samplers,
                .RegisterSpace = 1,
                .OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            };
            ++range_index;
        }

        const uint32_t num_root_params = num_desc_ranges + shader_rfl.num_cbvs;
        std::unique_ptr<D3D12_ROOT_PARAMETER[]> root_params;
        if (num_root_params > 0)
        {
            root_params = std::make_unique<D3D12_ROOT_PARAMETER[]>(num_root_params);
            uint32_t root_index = 0;
            range_index = 0;
            if (shader_rfl.num_srvs != 0)
            {
                root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[range_index], 1);
                ++root_index;
                ++range_index;
            }
            if (shader_rfl.num_uavs != 0)
            {
                root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[range_index], 1);
                ++root_index;
                ++range_index;
            }
            if (shader_rfl.num_dynamic_samplers != 0)
            {
                root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[range_index], 1);
                ++root_index;
                ++range_index;
            }
            for (uint32_t i = 0; i < shader_rfl.num_cbvs; ++i)
            {
                root_params[root_index] = CreateRootParameterAsConstantBufferView(i);
                ++root_index;
            }
        }

        auto d3d12_static_samplers = std::make_unique<D3D12_STATIC_SAMPLER_DESC[]>(static_samplers.size());
        for (uint32_t i = 0; i < static_samplers.size(); ++i)
        {
            d3d12_static_samplers[i] = D3D12Imp(static_samplers[i]).SamplerDesc(i);
        }

        const D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {
            .NumParameters = num_root_params,
            .pParameters = root_params.get(),
            .NumStaticSamplers = static_cast<uint32_t>(static_samplers.size()),
            .pStaticSamplers = d3d12_static_samplers.get(),
            .Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        HRESULT hr = ::D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.Put(), error.Put());
        if (FAILED(hr))
        {
            ::OutputDebugStringA(
                std::format("D3D12SerializeRootSignature failed: {}\n", static_cast<const char*>(error->GetBufferPointer())).c_str());
            TIFHR(hr);
        }

        ID3D12Device* d3d12_device = D3D12Imp(gpu_system).Device();

        TIFHR(d3d12_device->CreateRootSignature(
            1, blob->GetBufferPointer(), blob->GetBufferSize(), UuidOf<ID3D12RootSignature>(), root_sig_.Object().PutVoid()));

        const D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{
            .pRootSignature = root_sig_.Object().Get(),
            .CS{
                .pShaderBytecode = bytecode.data(),
                .BytecodeLength = bytecode.size(),
            },
            .NodeMask = 0,
        };
        TIFHR(d3d12_device->CreateComputePipelineState(&pso_desc, UuidOf<ID3D12PipelineState>(), pso_.Object().PutVoid()));
    }

    D3D12ComputePipeline::~D3D12ComputePipeline() = default;

    D3D12ComputePipeline::D3D12ComputePipeline(D3D12ComputePipeline&& other) noexcept = default;
    D3D12ComputePipeline::D3D12ComputePipeline(GpuComputePipelineInternal&& other) noexcept
        : D3D12ComputePipeline(static_cast<D3D12ComputePipeline&&>(other))
    {
    }

    D3D12ComputePipeline& D3D12ComputePipeline::operator=(D3D12ComputePipeline&& other) noexcept = default;
    GpuComputePipelineInternal& D3D12ComputePipeline::operator=(GpuComputePipelineInternal&& other) noexcept
    {
        return this->operator=(static_cast<D3D12ComputePipeline&&>(other));
    }

    void D3D12ComputePipeline::Bind(GpuCommandList& cmd_list) const
    {
        this->Bind(D3D12Imp(cmd_list));
    }

    void D3D12ComputePipeline::Bind(D3D12CommandList& cmd_list) const
    {
        auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

        d3d12_cmd_list->SetPipelineState(pso_.Object().Get());
        d3d12_cmd_list->SetComputeRootSignature(root_sig_.Object().Get());
    }

    const D3D12BindingSlots& D3D12ComputePipeline::BindingSlots() const noexcept
    {
        return binding_slots_;
    }

    const std::string& D3D12ComputePipeline::ShaderName() const noexcept
    {
        return shader_name_;
    }
} // namespace AIHoloImager
