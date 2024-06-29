// Copyright (c) 2024 Minmin Gong
//

#include "MultiViewRenderer.hpp"

#include <format>

#include "Gpu/GpuBufferHelper.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuDescriptorAllocator.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuTexture.hpp"
#include "MvDiffusion/MultiViewDiffusion.hpp"
#include "Util/ComPtr.hpp"
#include "Util/ErrorHandling.hpp"
#include "Util/Uuid.hpp"

#include "CompiledShader/DownsampleCs.h"
#include "CompiledShader/RenderPs.h"
#include "CompiledShader/RenderVs.h"

using namespace DirectX;
using namespace AIHoloImager;

namespace
{
    // The angles are defined by zero123plus v1.2 (https://github.com/SUDO-AI-3D/zero123plus)
    constexpr float Azimuths[] = {30, 90, 150, 210, 270, 330};
    constexpr float Elevations[] = {20, -10, 20, -10, 20, -10};
    constexpr float Fov = XM_PI / 6;
    const float MvScale = 1.6f; // The fine-tuned zero123plus in InstantMesh has a scale
                                // (https://github.com/TencentARC/InstantMesh/commit/34c193cc96eebd46deb7c48a76613753ad777122)

    constexpr uint32_t DivUp(uint32_t a, uint32_t b) noexcept
    {
        return (a + b - 1) / b;
    }

    D3D12_ROOT_PARAMETER CreateRootParameterAsDescriptorTable(const D3D12_DESCRIPTOR_RANGE* descriptor_ranges,
        uint32_t num_descriptor_ranges, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        D3D12_ROOT_PARAMETER ret;
        ret.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        ret.DescriptorTable.NumDescriptorRanges = num_descriptor_ranges;
        ret.DescriptorTable.pDescriptorRanges = descriptor_ranges;
        ret.ShaderVisibility = visibility;
        return ret;
    }

    D3D12_ROOT_PARAMETER CreateRootParameterAsConstantBufferView(
        uint32_t shader_register, uint32_t register_space = 0, D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        D3D12_ROOT_PARAMETER ret;
        ret.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        ret.Descriptor.ShaderRegister = shader_register;
        ret.Descriptor.RegisterSpace = register_space;
        ret.ShaderVisibility = visibility;
        return ret;
    }

    void UploadGpuTexture(GpuSystem& gpu_system, const Texture& tex, GpuTexture2D& output_tex)
    {
        if (!output_tex || (output_tex.Width(0) != tex.Width()) || (output_tex.Height(0) != tex.Height()))
        {
            output_tex = GpuTexture2D(gpu_system, tex.Width(), tex.Height(), 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        }

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);
        output_tex.Upload(gpu_system, cmd_list, 0, tex.Data());
        gpu_system.Execute(std::move(cmd_list));
    }

    Texture ReadbackGpuTexture(GpuSystem& gpu_system, const GpuTexture2D& texture)
    {
        Texture ret(texture.Width(0), texture.Height(0), FormatSize(texture.Format()));

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Render);
        texture.Readback(gpu_system, cmd_list, 0, ret.Data());
        gpu_system.Execute(std::move(cmd_list));

        return ret;
    }

    void RemoveAlpha(Texture& tex)
    {
        if (tex.NumChannels() != 3)
        {
            Texture ret(tex.Width(), tex.Height(), 3);

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < tex.Width() * tex.Height(); ++i)
            {
                memmove(&ret.Data()[i * 3], &tex.Data()[i * tex.NumChannels()], 3);
            }

            tex = std::move(ret);
        }
    }

    XMVECTOR SphericalCameraPose(float azimuth_deg, float elevation_deg, float radius)
    {
        const float azimuth = XMConvertToRadians(azimuth_deg);
        const float elevation = XMConvertToRadians(elevation_deg);

        float sin_azimuth;
        float cos_azimuth;
        XMScalarSinCos(&sin_azimuth, &cos_azimuth, azimuth);

        float sin_elevation;
        float cos_elevation;
        XMScalarSinCos(&sin_elevation, &cos_elevation, elevation);

        const float x = cos_elevation * cos_azimuth;
        const float y = sin_elevation;
        const float z = cos_elevation * sin_azimuth;
        return XMVectorSet(x, y, z, 0) * radius;
    }
} // namespace

namespace AIHoloImager
{
    class MultiViewRenderer::Impl
    {
    public:
        Impl(GpuSystem& gpu_system, PythonSystem& python_system, uint32_t width, uint32_t height)
            : gpu_system_(gpu_system), python_system_(python_system), proj_mtx_(XMMatrixPerspectiveFovLH(Fov, 1, 0.1f, 30))
        {
            rtv_desc_block_ = gpu_system_.AllocRtvDescBlock(1);
            rtv_descriptor_size_ = gpu_system_.RtvDescSize();

            dsv_desc_block_ = gpu_system_.AllocDsvDescBlock(1);
            dsv_descriptor_size_ = gpu_system_.DsvDescSize();

            const DXGI_FORMAT color_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
            const DXGI_FORMAT ds_fmt = DXGI_FORMAT_D32_FLOAT;

            ssaa_rt_tex_ = GpuTexture2D(gpu_system_, width * SsaaScale, height * SsaaScale, 1, color_fmt,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON, L"ssaa_rt_tex_");
            ssaa_rtv_ = GpuRenderTargetView(
                gpu_system_, ssaa_rt_tex_, DXGI_FORMAT_UNKNOWN, OffsetHandle(rtv_desc_block_.CpuHandle(), 0, rtv_descriptor_size_));

            ssaa_ds_tex_ = GpuTexture2D(gpu_system_, width * SsaaScale, height * SsaaScale, 1, ds_fmt,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_COMMON, L"ssaa_ds_tex_");
            ssaa_dsv_ = GpuDepthStencilView(
                gpu_system_, ssaa_ds_tex_, DXGI_FORMAT_UNKNOWN, OffsetHandle(dsv_desc_block_.CpuHandle(), 0, dsv_descriptor_size_));

            init_view_tex_ = GpuTexture2D(gpu_system_, width, height, 1, color_fmt, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, L"init_view_tex_");
            for (size_t i = 0; i < std::size(multi_view_texs_); ++i)
            {
                multi_view_texs_[i] = GpuTexture2D(gpu_system_, width, height, 1, init_view_tex_.Format(),
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, std::format(L"multi_view_tex_{}", i));
            }

            ComPtr<ID3D12Device> d3d12_device(gpu_system_.NativeDevice());

            {
                render_cb_ = ConstantBuffer<RenderConstantBuffer>(gpu_system_, 1, L"render_cb_");
                render_srv_uav_desc_block_ = gpu_system_.AllocCbvSrvUavDescBlock(1);

                const D3D12_DESCRIPTOR_RANGE ranges[] = {
                    {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
                };

                const D3D12_ROOT_PARAMETER root_params[] = {
                    // VS
                    CreateRootParameterAsConstantBufferView(0),
                    // PS
                    CreateRootParameterAsDescriptorTable(&ranges[0], 1),
                };

                D3D12_STATIC_SAMPLER_DESC point_sampler_desc{};
                point_sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
                point_sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.MaxAnisotropy = 16;
                point_sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                point_sampler_desc.MinLOD = 0.0f;
                point_sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
                point_sampler_desc.ShaderRegister = 0;

                const D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {static_cast<uint32_t>(std::size(root_params)), root_params, 1,
                    &point_sampler_desc, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};

                ComPtr<ID3DBlob> blob;
                ComPtr<ID3DBlob> error;
                HRESULT hr = ::D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.Put(), error.Put());
                if (FAILED(hr))
                {
                    ::OutputDebugStringW(
                        std::format(L"D3D12SerializeRootSignature failed: {}\n", static_cast<const wchar_t*>(error->GetBufferPointer()))
                            .c_str());
                    TIFHR(hr);
                }

                TIFHR(d3d12_device->CreateRootSignature(
                    1, blob->GetBufferPointer(), blob->GetBufferSize(), UuidOf<ID3D12RootSignature>(), render_root_sig_.PutVoid()));

                const D3D12_INPUT_ELEMENT_DESC input_elems[] = {
                    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                };

                D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
                pso_desc.pRootSignature = render_root_sig_.Get();
                pso_desc.VS.pShaderBytecode = RenderVs_shader;
                pso_desc.VS.BytecodeLength = sizeof(RenderVs_shader);
                pso_desc.PS.pShaderBytecode = RenderPs_shader;
                pso_desc.PS.BytecodeLength = sizeof(RenderPs_shader);
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
                pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
                pso_desc.RasterizerState.FrontCounterClockwise = TRUE;
                pso_desc.RasterizerState.DepthClipEnable = TRUE;
                pso_desc.DepthStencilState.DepthEnable = TRUE;
                pso_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
                pso_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
                pso_desc.InputLayout.pInputElementDescs = input_elems;
                pso_desc.InputLayout.NumElements = static_cast<uint32_t>(std::size(input_elems));
                pso_desc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
                pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
                pso_desc.NumRenderTargets = 1;
                pso_desc.RTVFormats[0] = ssaa_rt_tex_.Format();
                pso_desc.DSVFormat = ssaa_ds_tex_.Format();
                pso_desc.SampleDesc.Count = 1;
                pso_desc.SampleDesc.Quality = 0;
                pso_desc.NodeMask = 0;
                pso_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
                TIFHR(d3d12_device->CreateGraphicsPipelineState(&pso_desc, UuidOf<ID3D12PipelineState>(), render_pso_.PutVoid()));
            }
            {
                downsample_srv_uav_desc_block_ = gpu_system_.AllocCbvSrvUavDescBlock(2);

                const D3D12_DESCRIPTOR_RANGE ranges[] = {
                    {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
                    {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
                };

                const D3D12_ROOT_PARAMETER root_params[] = {
                    CreateRootParameterAsDescriptorTable(&ranges[0], 1),
                    CreateRootParameterAsDescriptorTable(&ranges[1], 1),
                };

                const D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {
                    static_cast<uint32_t>(std::size(root_params)), root_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};

                ComPtr<ID3DBlob> blob;
                ComPtr<ID3DBlob> error;
                HRESULT hr = ::D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.Put(), error.Put());
                if (FAILED(hr))
                {
                    ::OutputDebugStringW(
                        std::format(L"D3D12SerializeRootSignature failed: {}\n", static_cast<const wchar_t*>(error->GetBufferPointer()))
                            .c_str());
                    TIFHR(hr);
                }

                TIFHR(d3d12_device->CreateRootSignature(
                    1, blob->GetBufferPointer(), blob->GetBufferSize(), UuidOf<ID3D12RootSignature>(), downsample_root_sig_.PutVoid()));

                D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc;
                pso_desc.pRootSignature = downsample_root_sig_.Get();
                pso_desc.CS.pShaderBytecode = DownsampleCs_shader;
                pso_desc.CS.BytecodeLength = sizeof(DownsampleCs_shader);
                pso_desc.NodeMask = 0;
                pso_desc.CachedPSO.pCachedBlob = nullptr;
                pso_desc.CachedPSO.CachedBlobSizeInBytes = 0;
                pso_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
                TIFHR(d3d12_device->CreateComputePipelineState(&pso_desc, UuidOf<ID3D12PipelineState>(), downsample_pso_.PutVoid()));
            }
        }

        ~Impl() noexcept
        {
            render_cb_ = ConstantBuffer<RenderConstantBuffer>();
            render_root_sig_ = nullptr;
            render_pso_ = nullptr;
            gpu_system_.DeallocCbvSrvUavDescBlock(std::move(render_srv_uav_desc_block_));

            ssaa_dsv_.Reset();
            ssaa_ds_tex_.Reset();

            ssaa_rtv_.Reset();
            ssaa_rt_tex_.Reset();

            init_view_tex_.Reset();
            for (auto& tex : multi_view_texs_)
            {
                tex.Reset();
            }

            gpu_system_.DeallocDsvDescBlock(std::move(dsv_desc_block_));
            gpu_system_.DeallocRtvDescBlock(std::move(rtv_desc_block_));

            gpu_system_.WaitForGpu();
        }

        Result Render(const Mesh& mesh)
        {
            GpuBuffer vb(gpu_system_, static_cast<uint32_t>(mesh.Vertices().size() * sizeof(Mesh::VertexFormat)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"vb");
            memcpy(vb.Map(), mesh.Vertices().data(), vb.Size());
            vb.Unmap(D3D12_RANGE{0, vb.Size()});

            GpuBuffer ib(gpu_system_, static_cast<uint32_t>(mesh.Indices().size() * sizeof(uint32_t)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"ib");
            memcpy(ib.Map(), mesh.Indices().data(), ib.Size());
            ib.Unmap(D3D12_RANGE{0, ib.Size()});

            GpuTexture2D diffuse_tex;
            UploadGpuTexture(gpu_system_, mesh.AlbedoTexture(), diffuse_tex);

            GpuShaderResourceView diffuse_srv(gpu_system_, diffuse_tex, render_srv_uav_desc_block_.CpuHandle());

            constexpr float CameraDist = 10;

            const uint32_t num_indices = static_cast<uint32_t>(mesh.Indices().size());

            {
                GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                RenderToSsaa(cmd_list, vb, ib, num_indices, 0, 45, CameraDist);
                Downsample(cmd_list, init_view_tex_);
                gpu_system_.Execute(std::move(cmd_list));
                gpu_system_.WaitForGpu();
            }

            Texture init_view_cpu_tex = ReadbackGpuTexture(gpu_system_, init_view_tex_);
            RemoveAlpha(init_view_cpu_tex);

            MultiViewDiffusion mv_diffusion(python_system_);
            Texture mv_diffusion_tex = mv_diffusion.Generate(init_view_cpu_tex);

            for (size_t i = 0; i < std::size(Azimuths); ++i)
            {
                {
                    GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                    RenderToSsaa(cmd_list, vb, ib, num_indices, Azimuths[i], Elevations[i], CameraDist, MvScale);
                    gpu_system_.Execute(std::move(cmd_list));
                    gpu_system_.WaitForGpu();
                }
                BlendWithDiffusion(mv_diffusion_tex, static_cast<uint32_t>(i));
                {
                    GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                    Downsample(cmd_list, multi_view_texs_[i]);
                    gpu_system_.Execute(std::move(cmd_list));
                    gpu_system_.WaitForGpu();
                }
            }

            Result ret;
            for (uint32_t i = 0; i < 6; ++i)
            {
                ret.multi_view_images[i] = ReadbackGpuTexture(gpu_system_, multi_view_texs_[i]);
                RemoveAlpha(ret.multi_view_images[i]);
            }

            return ret;
        }

    private:
        void RenderToSsaa(GpuCommandList& cmd_list, GpuBuffer& vb, GpuBuffer& ib, uint32_t num_indices, float camera_azimuth,
            float camera_elevation, float camera_dist, float scale = 1)
        {
            auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

            const XMVECTOR camera_pos = SphericalCameraPose(camera_azimuth, camera_elevation, camera_dist);
            const XMVECTOR camera_dir = -XMVector3Normalize(camera_pos);
            XMVECTOR up_vec;
            if (-XMVectorGetY(camera_dir) > 0.95f)
            {
                up_vec = XMVectorSet(1, 0, 0, 0);
            }
            else
            {
                up_vec = XMVectorSet(0, 1, 0, 0);
            }

            const XMMATRIX view_mtx = XMMatrixLookAtLH(camera_pos, XMVectorSet(0, 0, 0, 1.0f), up_vec);

            XMStoreFloat4x4(&render_cb_->mvp, XMMatrixTranspose(view_mtx * proj_mtx_));
            render_cb_.UploadToGpu();

            vb.Transition(cmd_list, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            ib.Transition(cmd_list, D3D12_RESOURCE_STATE_INDEX_BUFFER);

            ssaa_ds_tex_.Transition(cmd_list, D3D12_RESOURCE_STATE_DEPTH_WRITE);

            D3D12_VERTEX_BUFFER_VIEW vbv{};
            vbv.BufferLocation = vb.GpuVirtualAddress();
            vbv.SizeInBytes = vb.Size();
            vbv.StrideInBytes = sizeof(Mesh::VertexFormat);
            d3d12_cmd_list->IASetVertexBuffers(0, 1, &vbv);

            D3D12_INDEX_BUFFER_VIEW ibv{};
            ibv.BufferLocation = ib.GpuVirtualAddress();
            ibv.SizeInBytes = ib.Size();
            ibv.Format = DXGI_FORMAT_R32_UINT;
            d3d12_cmd_list->IASetIndexBuffer(&ibv);

            d3d12_cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            d3d12_cmd_list->SetPipelineState(render_pso_.Get());
            d3d12_cmd_list->SetGraphicsRootSignature(render_root_sig_.Get());

            ID3D12DescriptorHeap* heaps[] = {render_srv_uav_desc_block_.NativeDescriptorHeap()};
            d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);
            d3d12_cmd_list->SetGraphicsRootDescriptorTable(1, render_srv_uav_desc_block_.GpuHandle());

            d3d12_cmd_list->SetGraphicsRootConstantBufferView(0, render_cb_.GpuVirtualAddress());

            ssaa_rt_tex_.Transition(cmd_list, D3D12_RESOURCE_STATE_RENDER_TARGET);

            D3D12_CPU_DESCRIPTOR_HANDLE rtvs[] = {ssaa_rtv_.CpuHandle()};
            D3D12_CPU_DESCRIPTOR_HANDLE dsv = ssaa_dsv_.CpuHandle();
            d3d12_cmd_list->OMSetRenderTargets(static_cast<uint32_t>(std::size(rtvs)), rtvs, true, &dsv);

            float clear_clr[] = {0, 0, 0, 0};
            d3d12_cmd_list->ClearRenderTargetView(rtvs[0], clear_clr, 0, nullptr);
            d3d12_cmd_list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1, 0, 0, nullptr);

            const float offset_scale = (scale - 1) / 2;
            D3D12_VIEWPORT viewport{-offset_scale * ssaa_rt_tex_.Width(0), -offset_scale * ssaa_rt_tex_.Height(0),
                scale * ssaa_rt_tex_.Width(0), scale * ssaa_rt_tex_.Height(0), 0, 1};
            d3d12_cmd_list->RSSetViewports(1, &viewport);

            D3D12_RECT scissor_rc{0, 0, static_cast<LONG>(ssaa_rt_tex_.Width(0)), static_cast<LONG>(ssaa_rt_tex_.Height(0))};
            d3d12_cmd_list->RSSetScissorRects(1, &scissor_rc);

            d3d12_cmd_list->DrawIndexedInstanced(num_indices, 1, 0, 0, 0);

            ssaa_rt_tex_.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
        }

        void Downsample(GpuCommandList& cmd_list, GpuTexture2D& target)
        {
            auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

            uint32_t descriptor_size = gpu_system_.CbvSrvUavDescSize();

            d3d12_cmd_list->SetPipelineState(downsample_pso_.Get());
            d3d12_cmd_list->SetComputeRootSignature(downsample_root_sig_.Get());

            GpuShaderResourceView srv(
                gpu_system_, ssaa_rt_tex_, OffsetHandle(downsample_srv_uav_desc_block_.CpuHandle(), 0, descriptor_size));
            GpuUnorderedAccessView uav(gpu_system_, target, OffsetHandle(downsample_srv_uav_desc_block_.CpuHandle(), 1, descriptor_size));

            ID3D12DescriptorHeap* heaps[] = {downsample_srv_uav_desc_block_.NativeDescriptorHeap()};
            d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);
            d3d12_cmd_list->SetComputeRootDescriptorTable(0, OffsetHandle(downsample_srv_uav_desc_block_.GpuHandle(), 0, descriptor_size));
            d3d12_cmd_list->SetComputeRootDescriptorTable(1, OffsetHandle(downsample_srv_uav_desc_block_.GpuHandle(), 1, descriptor_size));

            target.Transition(cmd_list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            constexpr uint32_t BlockDim = 16;
            d3d12_cmd_list->Dispatch(DivUp(target.Width(0), BlockDim), DivUp(target.Height(0), BlockDim), 1);

            target.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
            ssaa_rt_tex_.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
        }

        void BlendWithDiffusion(Texture& mv_diffusion_tex, uint32_t index)
        {
            // TODO #13: Port to GPU

            Texture rendered_read_back_tex = ReadbackGpuTexture(gpu_system_, ssaa_rt_tex_);

            XMUINT2 rendered_min(rendered_read_back_tex.Width(), rendered_read_back_tex.Height());
            XMUINT2 rendered_max(0, 0);
            uint8_t* rendered_data = rendered_read_back_tex.Data();
            const uint32_t rendered_width = rendered_read_back_tex.Width();
            const uint32_t rendered_channels = rendered_read_back_tex.NumChannels();
            for (uint32_t y = 0; y < rendered_read_back_tex.Height(); ++y)
            {
                for (uint32_t x = 0; x < rendered_read_back_tex.Width(); ++x)
                {
                    if (rendered_data[(y * rendered_width + x) * rendered_channels + 3] != 0)
                    {
                        rendered_min.x = std::min(rendered_min.x, x);
                        rendered_max.x = std::max(rendered_max.x, x);
                        rendered_min.y = std::min(rendered_min.y, y);
                        rendered_max.y = std::max(rendered_max.y, y);
                    }
                }
            }

            constexpr uint32_t AtlasWidth = 2;
            constexpr uint32_t AtlasHeight = 3;
            constexpr uint32_t ValidThreshold = 237;

            const uint32_t view_width = mv_diffusion_tex.Width() / AtlasWidth;
            const uint32_t view_height = mv_diffusion_tex.Height() / AtlasHeight;

            const uint32_t atlas_y = index / AtlasWidth;
            const uint32_t atlas_x = index - atlas_y * AtlasWidth;
            const uint32_t atlas_offset_x = atlas_x * view_width;
            const uint32_t atlas_offset_y = atlas_y * view_height;

            XMUINT2 diffusion_min(view_width, view_height);
            XMUINT2 diffusion_max(0, 0);
            const uint8_t* diffusion_data = mv_diffusion_tex.Data();
            const uint32_t diffusion_width = mv_diffusion_tex.Width();
            const uint32_t diffusion_channels = mv_diffusion_tex.NumChannels();
            for (uint32_t y = 0; y < view_height; ++y)
            {
                for (uint32_t x = 0; x < view_width; ++x)
                {
                    const uint32_t pixel_offset = ((atlas_offset_y + y) * diffusion_width + (atlas_offset_x + x)) * diffusion_channels;
                    for (uint32_t c = 0; c < 3; ++c)
                    {
                        if (diffusion_data[pixel_offset + c] < ValidThreshold)
                        {
                            diffusion_min.x = std::min(diffusion_min.x, x);
                            diffusion_max.x = std::max(diffusion_max.x, x);
                            diffusion_min.y = std::min(diffusion_min.y, y);
                            diffusion_max.y = std::max(diffusion_max.y, y);
                            break;
                        }
                    }
                }
            }

            const float scale_x = static_cast<float>(diffusion_max.x - diffusion_min.x) / (rendered_max.x - rendered_min.x);
            const float scale_y = static_cast<float>(diffusion_max.y - diffusion_min.y) / (rendered_max.y - rendered_min.y);
            const float scale = std::min(scale_x, scale_y);

            const auto is_empty = [](const uint8_t* rgb) -> bool {
                constexpr int empty_color_r = 0xFF;
                constexpr int empty_color_g = 0x7F;
                constexpr int empty_color_b = 0x27;
                return ((std::abs(static_cast<int>(rgb[0]) - empty_color_r) < 2) &&
                        (std::abs(static_cast<int>(rgb[1]) - empty_color_g) < 20) &&
                        (std::abs(static_cast<int>(rgb[2]) - empty_color_b) < 15));
            };

            const int rendered_center_x = rendered_read_back_tex.Width() / 2;
            const int rendered_center_y = rendered_read_back_tex.Height() / 2;
            const int diffusion_center_x = view_width / 2;
            const int diffusion_center_y = view_height / 2;
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t y = 0; y < rendered_read_back_tex.Height(); ++y)
            {
                const uint32_t src_y = static_cast<uint32_t>(
                    std::clamp(static_cast<int>(std::round((static_cast<int>(y) - rendered_center_y) * scale)) + diffusion_center_y, 0,
                        static_cast<int>(view_height - 1)));
                for (uint32_t x = 0; x < rendered_read_back_tex.Width(); ++x)
                {
                    const uint32_t src_x = static_cast<uint32_t>(
                        std::clamp(static_cast<int>(std::round((static_cast<int>(x) - rendered_center_x) * scale)) + diffusion_center_x, 0,
                            static_cast<int>(view_width - 1)));

                    const uint32_t diffusion_offset = (atlas_offset_y + src_y) * diffusion_width + (atlas_offset_x + src_x);
                    const uint32_t rendered_offset = y * rendered_width + x;

                    bool rendered_valid = false;
                    if ((rendered_data[rendered_offset * rendered_channels + 3] != 0) &&
                        !is_empty(&rendered_data[rendered_offset * rendered_channels]))
                    {
                        rendered_valid = true;
                    }

                    bool diffusion_valid = false;
                    for (uint32_t c = 0; c < 3; ++c)
                    {
                        if (diffusion_data[diffusion_offset * diffusion_channels + c] < ValidThreshold)
                        {
                            diffusion_valid = true;
                            break;
                        }
                    }

                    if (!rendered_valid && diffusion_valid)
                    {
                        memcpy(&rendered_data[rendered_offset * rendered_channels], &diffusion_data[diffusion_offset * diffusion_channels],
                            diffusion_channels);
                        rendered_data[rendered_offset * rendered_channels + 3] = 0xFF;
                    }
                }
            }

            UploadGpuTexture(gpu_system_, rendered_read_back_tex, ssaa_rt_tex_);
        }

    private:
        GpuSystem& gpu_system_;
        PythonSystem& python_system_;

        GpuDescriptorBlock rtv_desc_block_;
        uint32_t rtv_descriptor_size_;

        GpuDescriptorBlock dsv_desc_block_;
        uint32_t dsv_descriptor_size_;

        GpuTexture2D init_view_tex_;
        GpuTexture2D multi_view_texs_[6];

        static constexpr uint32_t SsaaScale = 4;

        GpuTexture2D ssaa_rt_tex_;
        GpuRenderTargetView ssaa_rtv_;

        GpuTexture2D ssaa_ds_tex_;
        GpuDepthStencilView ssaa_dsv_;

        XMMATRIX proj_mtx_;

        struct RenderConstantBuffer
        {
            DirectX::XMFLOAT4X4 mvp;
        };
        ConstantBuffer<RenderConstantBuffer> render_cb_;
        ComPtr<ID3D12RootSignature> render_root_sig_;
        ComPtr<ID3D12PipelineState> render_pso_;
        GpuDescriptorBlock render_srv_uav_desc_block_;

        ComPtr<ID3D12RootSignature> downsample_root_sig_;
        ComPtr<ID3D12PipelineState> downsample_pso_;
        GpuDescriptorBlock downsample_srv_uav_desc_block_;
    };

    MultiViewRenderer::MultiViewRenderer(GpuSystem& gpu_system, PythonSystem& python_system, uint32_t width, uint32_t height)
        : impl_(std::make_unique<Impl>(gpu_system, python_system, width, height))
    {
    }

    MultiViewRenderer::~MultiViewRenderer() noexcept = default;

    MultiViewRenderer::MultiViewRenderer(MultiViewRenderer&& other) noexcept = default;
    MultiViewRenderer& MultiViewRenderer::operator=(MultiViewRenderer&& other) noexcept = default;

    MultiViewRenderer::Result MultiViewRenderer::Render(const Mesh& mesh)
    {
        return impl_->Render(mesh);
    }
} // namespace AIHoloImager
