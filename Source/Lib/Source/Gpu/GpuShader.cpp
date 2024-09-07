// Copyright (c) 2024 Minmin Gong
//

#include "GpuShader.hpp"

#include <format>
#include <memory>

#include "Util/ErrorHandling.hpp"

namespace AIHoloImager
{
    GpuRenderPipeline::GpuRenderPipeline() noexcept = default;
    GpuRenderPipeline::GpuRenderPipeline(GpuSystem& gpu_system, const ShaderInfo shaders[NumShaderStages],
        std::span<const D3D12_INPUT_ELEMENT_DESC> input_elems, std::span<const D3D12_STATIC_SAMPLER_DESC> samplers, const States& states)
        : gpu_system_(&gpu_system)
    {
        uint32_t num_desc_ranges = 0;
        for (uint32_t s = 0; s < NumShaderStages; ++s)
        {
            const auto& shader = shaders[s];
            num_desc_ranges += (shader.num_srvs ? 1 : 0) + (shader.num_uavs ? 1 : 0);
        }

        auto ranges = std::make_unique<D3D12_DESCRIPTOR_RANGE[]>(num_desc_ranges);
        uint32_t range_index = 0;
        for (uint32_t s = 0; s < NumShaderStages; ++s)
        {
            const auto& shader = shaders[s];
            if (shader.num_srvs != 0)
            {
                ranges[range_index] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, shader.num_srvs, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
                ++range_index;
            }
            if (shader.num_uavs != 0)
            {
                ranges[range_index] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, shader.num_uavs, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
                ++range_index;
            }
        }

        uint32_t num_root_params = num_desc_ranges;
        for (uint32_t s = 0; s < NumShaderStages; ++s)
        {
            const auto& shader = shaders[s];
            num_root_params += shader.num_cbs;
        }

        std::unique_ptr<D3D12_ROOT_PARAMETER[]> root_params;
        if (num_root_params > 0)
        {
            root_params = std::make_unique<D3D12_ROOT_PARAMETER[]>(num_root_params);
            uint32_t root_index = 0;
            range_index = 0;
            for (uint32_t s = 0; s < NumShaderStages; ++s)
            {
                const auto& shader = shaders[s];

                D3D12_SHADER_VISIBILITY visibility;
                switch (static_cast<ShaderStage>(s))
                {
                case ShaderStage::Vertex:
                    visibility = D3D12_SHADER_VISIBILITY_VERTEX;
                    break;

                case ShaderStage::Pixel:
                    visibility = D3D12_SHADER_VISIBILITY_PIXEL;
                    break;

                default:
                    visibility = D3D12_SHADER_VISIBILITY_ALL;
                    break;
                }

                if (shader.num_srvs != 0)
                {
                    root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[range_index], 1, visibility);
                    ++root_index;
                    ++range_index;
                }
                if (shader.num_uavs != 0)
                {
                    root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[range_index], 1, visibility);
                    ++root_index;
                    ++range_index;
                }
                for (uint32_t i = 0; i < shader.num_cbs; ++i)
                {
                    root_params[root_index] = CreateRootParameterAsConstantBufferView(i, 0, visibility);
                    ++root_index;
                }
            }
        }

        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        if (!input_elems.empty())
        {
            flags |= D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        }

        const D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {
            num_root_params, root_params.get(), static_cast<uint32_t>(samplers.size()), samplers.data(), flags};

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        HRESULT hr = ::D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.Put(), error.Put());
        if (FAILED(hr))
        {
            ::OutputDebugStringA(
                std::format("D3D12SerializeRootSignature failed: {}\n", static_cast<const char*>(error->GetBufferPointer())).c_str());
            TIFHR(hr);
        }

        ID3D12Device* d3d12_device = gpu_system.NativeDevice();

        TIFHR(d3d12_device->CreateRootSignature(
            1, blob->GetBufferPointer(), blob->GetBufferSize(), UuidOf<ID3D12RootSignature>(), root_sig_.PutVoid()));

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root_sig_.Get();
        for (uint32_t s = 0; s < NumShaderStages; ++s)
        {
            const auto& shader = shaders[s];
            switch (static_cast<ShaderStage>(s))
            {
            case ShaderStage::Vertex:
                pso_desc.VS.pShaderBytecode = shader.bytecode.data();
                pso_desc.VS.BytecodeLength = shader.bytecode.size();
                break;

            case ShaderStage::Pixel:
                pso_desc.PS.pShaderBytecode = shader.bytecode.data();
                pso_desc.PS.BytecodeLength = shader.bytecode.size();
                break;

            default:
                throw std::runtime_error("Not supported yet");
            }
        }
        for (uint32_t i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
        {
            pso_desc.BlendState.RenderTarget[i].SrcBlend = D3D12_BLEND_ONE;
            pso_desc.BlendState.RenderTarget[i].DestBlend = D3D12_BLEND_ZERO;
            pso_desc.BlendState.RenderTarget[i].BlendOp = D3D12_BLEND_OP_ADD;
            pso_desc.BlendState.RenderTarget[i].SrcBlendAlpha = D3D12_BLEND_ONE;
            pso_desc.BlendState.RenderTarget[i].DestBlendAlpha = D3D12_BLEND_ZERO;
            pso_desc.BlendState.RenderTarget[i].BlendOpAlpha = D3D12_BLEND_OP_ADD;
            pso_desc.BlendState.RenderTarget[i].LogicOp = D3D12_LOGIC_OP_NOOP;
            pso_desc.BlendState.RenderTarget[i].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        }
        pso_desc.SampleMask = UINT_MAX;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        switch (states.cull_mode)
        {
        case CullMode::None:
            pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            break;
        case CullMode::ClockWise:
            pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
            break;
        case CullMode::CounterClockWise:
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
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = static_cast<uint32_t>(states.rtv_formats.size());
        for (size_t i = 0; i < states.rtv_formats.size(); ++i)
        {
            pso_desc.RTVFormats[i] = states.rtv_formats[i];
        }
        pso_desc.DSVFormat = states.dsv_format;
        pso_desc.SampleDesc.Count = 1;
        pso_desc.SampleDesc.Quality = 0;
        pso_desc.NodeMask = 0;
        pso_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        TIFHR(d3d12_device->CreateGraphicsPipelineState(&pso_desc, UuidOf<ID3D12PipelineState>(), pso_.PutVoid()));
    }

    GpuRenderPipeline::~GpuRenderPipeline()
    {
        if (root_sig_)
        {
            gpu_system_->Recycle(std::move(root_sig_));

            assert(pso_);
            gpu_system_->Recycle(std::move(pso_));
        }
    }

    GpuRenderPipeline::GpuRenderPipeline(GpuRenderPipeline&& other) noexcept = default;
    GpuRenderPipeline& GpuRenderPipeline::operator=(GpuRenderPipeline&& other) noexcept = default;

    ID3D12RootSignature* GpuRenderPipeline::NativeRootSignature() const noexcept
    {
        return root_sig_.Get();
    }

    ID3D12PipelineState* GpuRenderPipeline::NativePipelineState() const noexcept
    {
        return pso_.Get();
    }


    GpuComputePipeline::GpuComputePipeline() noexcept = default;
    GpuComputePipeline::GpuComputePipeline(
        GpuSystem& gpu_system, const ShaderInfo& shader, std::span<const D3D12_STATIC_SAMPLER_DESC> samplers)
        : gpu_system_(&gpu_system)
    {
        const uint32_t num_desc_ranges = (shader.num_srvs ? 1 : 0) + (shader.num_uavs ? 1 : 0);
        auto ranges = std::make_unique<D3D12_DESCRIPTOR_RANGE[]>(num_desc_ranges);
        uint32_t range_index = 0;
        if (shader.num_srvs != 0)
        {
            ranges[range_index] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, shader.num_srvs, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            ++range_index;
        }
        if (shader.num_uavs != 0)
        {
            ranges[range_index] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, shader.num_uavs, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            ++range_index;
        }

        const uint32_t num_root_params = num_desc_ranges + shader.num_cbs;
        std::unique_ptr<D3D12_ROOT_PARAMETER[]> root_params;
        if (num_root_params > 0)
        {
            root_params = std::make_unique<D3D12_ROOT_PARAMETER[]>(num_root_params);
            uint32_t root_index = 0;
            range_index = 0;
            if (shader.num_srvs != 0)
            {
                root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[range_index], 1);
                ++root_index;
                ++range_index;
            }
            if (shader.num_uavs != 0)
            {
                root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[range_index], 1);
                ++root_index;
                ++range_index;
            }
            for (uint32_t i = 0; i < shader.num_cbs; ++i)
            {
                root_params[root_index] = CreateRootParameterAsConstantBufferView(i);
                ++root_index;
            }
        }

        const D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {
            num_root_params, root_params.get(), static_cast<uint32_t>(samplers.size()), samplers.data(), D3D12_ROOT_SIGNATURE_FLAG_NONE};

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        HRESULT hr = ::D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.Put(), error.Put());
        if (FAILED(hr))
        {
            ::OutputDebugStringA(
                std::format("D3D12SerializeRootSignature failed: {}\n", static_cast<const char*>(error->GetBufferPointer())).c_str());
            TIFHR(hr);
        }

        ID3D12Device* d3d12_device = gpu_system.NativeDevice();

        TIFHR(d3d12_device->CreateRootSignature(
            1, blob->GetBufferPointer(), blob->GetBufferSize(), UuidOf<ID3D12RootSignature>(), root_sig_.PutVoid()));

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root_sig_.Get();
        pso_desc.CS.pShaderBytecode = shader.bytecode.data();
        pso_desc.CS.BytecodeLength = shader.bytecode.size();
        pso_desc.NodeMask = 0;
        TIFHR(d3d12_device->CreateComputePipelineState(&pso_desc, UuidOf<ID3D12PipelineState>(), pso_.PutVoid()));
    }

    GpuComputePipeline::~GpuComputePipeline()
    {
        if (root_sig_)
        {
            gpu_system_->Recycle(std::move(root_sig_));

            assert(pso_);
            gpu_system_->Recycle(std::move(pso_));
        }
    }

    GpuComputePipeline::GpuComputePipeline(GpuComputePipeline&& other) noexcept = default;
    GpuComputePipeline& GpuComputePipeline::operator=(GpuComputePipeline&& other) noexcept = default;

    ID3D12RootSignature* GpuComputePipeline::NativeRootSignature() const noexcept
    {
        return root_sig_.Get();
    }

    ID3D12PipelineState* GpuComputePipeline::NativePipelineState() const noexcept
    {
        return pso_.Get();
    }
} // namespace AIHoloImager