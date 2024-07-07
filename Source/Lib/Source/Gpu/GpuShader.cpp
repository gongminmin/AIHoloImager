// Copyright (c) 2024 Minmin Gong
//

#include "GpuShader.hpp"

#include <format>
#include <memory>

#include "Util/ErrorHandling.hpp"

namespace AIHoloImager
{
    GpuComputeShader::GpuComputeShader() noexcept = default;
    GpuComputeShader::GpuComputeShader(GpuSystem& gpu_system, std::span<const uint8_t> bytecode, uint32_t num_cbs, uint32_t num_srvs,
        uint32_t num_uavs, std::span<const D3D12_STATIC_SAMPLER_DESC> samplers)
    {
        const D3D12_DESCRIPTOR_RANGE ranges[] = {
            {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, num_srvs, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
            {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, num_uavs, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
        };

        const uint32_t num_root_params = num_cbs + num_srvs + num_uavs;

        auto root_params = std::make_unique<D3D12_ROOT_PARAMETER[]>(num_root_params);
        uint32_t root_index = 0;
        if (num_srvs != 0)
        {
            root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[0], 1);
            ++root_index;
        }
        if (num_uavs != 0)
        {
            root_params[root_index] = CreateRootParameterAsDescriptorTable(&ranges[1], 1);
            ++root_index;
        }
        for (uint32_t i = 0; i < num_cbs; ++i, ++root_index)
        {
            root_params[root_index] = CreateRootParameterAsConstantBufferView(i);
        }

        const D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {
            num_root_params, root_params.get(), static_cast<uint32_t>(samplers.size()), samplers.data(), D3D12_ROOT_SIGNATURE_FLAG_NONE};

        ComPtr<ID3DBlob> blob;
        ComPtr<ID3DBlob> error;
        HRESULT hr = ::D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.Put(), error.Put());
        if (FAILED(hr))
        {
            ::OutputDebugStringW(
                std::format(L"D3D12SerializeRootSignature failed: {}\n", static_cast<const wchar_t*>(error->GetBufferPointer())).c_str());
            TIFHR(hr);
        }

        ID3D12Device* d3d12_device = gpu_system.NativeDevice();

        TIFHR(d3d12_device->CreateRootSignature(
            1, blob->GetBufferPointer(), blob->GetBufferSize(), UuidOf<ID3D12RootSignature>(), root_sig_.PutVoid()));

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc;
        pso_desc.pRootSignature = root_sig_.Get();
        pso_desc.CS.pShaderBytecode = bytecode.data();
        pso_desc.CS.BytecodeLength = bytecode.size();
        pso_desc.NodeMask = 0;
        pso_desc.CachedPSO.pCachedBlob = nullptr;
        pso_desc.CachedPSO.CachedBlobSizeInBytes = 0;
        pso_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        TIFHR(d3d12_device->CreateComputePipelineState(&pso_desc, UuidOf<ID3D12PipelineState>(), pso_.PutVoid()));
    }
    GpuComputeShader::~GpuComputeShader() noexcept = default;

    GpuComputeShader::GpuComputeShader(GpuComputeShader&& other) noexcept = default;
    GpuComputeShader& GpuComputeShader::operator=(GpuComputeShader&& other) noexcept = default;

    ID3D12RootSignature* GpuComputeShader::NativeRootSignature() const noexcept
    {
        return root_sig_.Get();
    }

    ID3D12PipelineState* GpuComputeShader::NativePipelineState() const noexcept
    {
        return pso_.Get();
    }
} // namespace AIHoloImager