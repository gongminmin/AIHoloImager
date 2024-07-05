// Copyright (c) 2024 Minmin Gong
//

#include "PostProcessor.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <set>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuTexture.hpp"

#include "CompiledShader/DilateCs.h"
#include "CompiledShader/RefillTexturePs.h"
#include "CompiledShader/RefillTextureVs.h"

using namespace AIHoloImager;
using namespace DirectX;

namespace
{
    void Ensure4Channel(Texture& tex)
    {
        const uint32_t channels = tex.NumChannels();
        if (channels != 4)
        {
            Texture ret(tex.Width(), tex.Height(), 3);

            const uint8_t* src = tex.Data();
            uint8_t* dst = ret.Data();

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < tex.Width() * tex.Height(); ++i)
            {
                memcpy(&dst[i * 4], &src[i * channels], channels);
                dst[i * 4 + 3] = 0xFF;
            }

            tex = std::move(ret);
        }
    }
} // namespace

namespace AIHoloImager
{
    class PostProcessor::Impl
    {
    public:
        Impl(const std::filesystem::path& exe_dir, GpuSystem& gpu_system) : exe_dir_(exe_dir), gpu_system_(gpu_system)
        {
            rtv_desc_block_ = gpu_system_.AllocRtvDescBlock(1);
            rtv_descriptor_size_ = gpu_system_.RtvDescSize();
            srv_descriptor_size_ = gpu_system.CbvSrvUavDescSize();

            ID3D12Device* d3d12_device = gpu_system_.NativeDevice();

            {
                refill_texture_srv_uav_desc_block_ = gpu_system_.AllocCbvSrvUavDescBlock(2);

                const D3D12_DESCRIPTOR_RANGE ranges[] = {
                    {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
                };

                const D3D12_ROOT_PARAMETER root_params[] = {
                    // PS
                    CreateRootParameterAsDescriptorTable(&ranges[0], 1),
                };

                D3D12_STATIC_SAMPLER_DESC bilinear_sampler_desc{};
                bilinear_sampler_desc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                bilinear_sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.MaxAnisotropy = 16;
                bilinear_sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                bilinear_sampler_desc.MinLOD = 0.0f;
                bilinear_sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
                bilinear_sampler_desc.ShaderRegister = 0;

                const D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {static_cast<uint32_t>(std::size(root_params)), root_params, 1,
                    &bilinear_sampler_desc, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};

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
                    1, blob->GetBufferPointer(), blob->GetBufferSize(), UuidOf<ID3D12RootSignature>(), refill_texture_root_sig_.PutVoid()));

                const D3D12_INPUT_ELEMENT_DESC input_elems[] = {
                    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                };

                D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
                pso_desc.pRootSignature = refill_texture_root_sig_.Get();
                pso_desc.VS.pShaderBytecode = RefillTextureVs_shader;
                pso_desc.VS.BytecodeLength = sizeof(RefillTextureVs_shader);
                pso_desc.PS.pShaderBytecode = RefillTexturePs_shader;
                pso_desc.PS.BytecodeLength = sizeof(RefillTexturePs_shader);
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
                pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
                pso_desc.RasterizerState.FrontCounterClockwise = TRUE;
                pso_desc.RasterizerState.DepthClipEnable = TRUE;
                pso_desc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON;
                pso_desc.DepthStencilState.DepthEnable = FALSE;
                pso_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
                pso_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
                pso_desc.InputLayout.pInputElementDescs = input_elems;
                pso_desc.InputLayout.NumElements = static_cast<uint32_t>(std::size(input_elems));
                pso_desc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
                pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
                pso_desc.NumRenderTargets = 1;
                pso_desc.RTVFormats[0] = ColorFmt;
                pso_desc.DSVFormat = DXGI_FORMAT_UNKNOWN;
                pso_desc.SampleDesc.Count = 1;
                pso_desc.SampleDesc.Quality = 0;
                pso_desc.NodeMask = 0;
                pso_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
                TIFHR(d3d12_device->CreateGraphicsPipelineState(&pso_desc, UuidOf<ID3D12PipelineState>(), refill_texture_pso_.PutVoid()));
            }
            {
                dilate_srv_uav_desc_block_ = gpu_system_.AllocCbvSrvUavDescBlock(2 * DilateTimes);

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
                    1, blob->GetBufferPointer(), blob->GetBufferSize(), UuidOf<ID3D12RootSignature>(), dilate_root_sig_.PutVoid()));

                D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc;
                pso_desc.pRootSignature = dilate_root_sig_.Get();
                pso_desc.CS.pShaderBytecode = DilateCs_shader;
                pso_desc.CS.BytecodeLength = sizeof(DilateCs_shader);
                pso_desc.NodeMask = 0;
                pso_desc.CachedPSO.pCachedBlob = nullptr;
                pso_desc.CachedPSO.CachedBlobSizeInBytes = 0;
                pso_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
                TIFHR(d3d12_device->CreateComputePipelineState(&pso_desc, UuidOf<ID3D12PipelineState>(), dilate_pso_.PutVoid()));
            }
        }

        ~Impl()
        {
            refill_texture_root_sig_ = nullptr;
            refill_texture_pso_ = nullptr;
            gpu_system_.DeallocCbvSrvUavDescBlock(std::move(refill_texture_srv_uav_desc_block_));
            gpu_system_.DeallocRtvDescBlock(std::move(rtv_desc_block_));

            gpu_system_.WaitForGpu();
        }

        Mesh Process(const MeshReconstruction::Result& recon_input, const Mesh& ai_mesh, const std::filesystem::path& tmp_dir)
        {
            const XMMATRIX transform_mtx = XMLoadFloat4x4(&recon_input.transform);

            std::vector<XMFLOAT3> rh_positions(ai_mesh.Vertices().size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(ai_mesh.Vertices().size()); ++i)
            {
                XMFLOAT3 pos = ai_mesh.Vertex(i).pos;
                std::swap(pos.y, pos.z);

                XMStoreFloat3(&pos, XMVector3TransformCoord(XMLoadFloat3(&pos), transform_mtx));
                pos.z = -pos.z;

                rh_positions[i] = pos;
            }

            DirectX::BoundingOrientedBox ai_obb;
            BoundingOrientedBox::CreateFromPoints(ai_obb, rh_positions.size(), &rh_positions[0], sizeof(rh_positions[0]));

            const float scale_x = ai_obb.Extents.x / recon_input.obb.Extents.x;
            const float scale_y = ai_obb.Extents.y / recon_input.obb.Extents.y;
            const float scale_z = ai_obb.Extents.z / recon_input.obb.Extents.z;
            const float scale = 1 / std::max({scale_x, scale_y, scale_z});

            Mesh transformed_mesh(static_cast<uint32_t>(ai_mesh.Vertices().size()), static_cast<uint32_t>(ai_mesh.Indices().size()));

            for (uint32_t i = 0; i < static_cast<uint32_t>(transformed_mesh.Vertices().size()); ++i)
            {
                auto& vertex = transformed_mesh.Vertex(i);

                XMFLOAT3 pos = ai_mesh.Vertex(i).pos;
                pos.x *= scale;
                pos.y *= scale;
                pos.z *= scale;
                std::swap(pos.y, pos.z);

                XMStoreFloat3(&pos, XMVector3TransformCoord(XMLoadFloat3(&pos), transform_mtx));
                pos.z = -pos.z;

                vertex.pos = pos;
                vertex.texcoord = XMFLOAT2(0, 0); // TextureMesh can't handle mesh with texture coordinate. Clear it.
            }

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < static_cast<uint32_t>(transformed_mesh.Indices().size()); ++i)
            {
                transformed_mesh.Index(i) = ai_mesh.Index(i);
            }

            const auto working_dir = tmp_dir / "Mvs";
            std::filesystem::create_directories(working_dir);

            const std::string tmp_mesh_name = "Temp_Ai";
            SaveMesh(transformed_mesh, working_dir / (tmp_mesh_name + ".glb"));

            const std::string mesh_name = this->MeshTexturing("Temp", tmp_mesh_name, working_dir);

            Mesh textured_mesh = LoadMesh(working_dir / (mesh_name + ".glb"));

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < static_cast<uint32_t>(transformed_mesh.Vertices().size()); ++i)
            {
                transformed_mesh.Vertex(i).texcoord = ai_mesh.Vertex(i).texcoord;
            }
            RefillTexture(transformed_mesh, ai_mesh, textured_mesh);

            const XMVECTOR center = XMLoadFloat3(&recon_input.obb.Center);
            const XMMATRIX pre_trans = XMMatrixTranslationFromVector(-center);
            const XMMATRIX pre_rotate =
                XMMatrixRotationQuaternion(XMQuaternionInverse(XMLoadFloat4(&recon_input.obb.Orientation))) * XMMatrixRotationZ(XM_PI / 2);
            const XMMATRIX pre_scale = XMMatrixScaling(1, -1, -1);

            const XMMATRIX adjust_mtx = pre_trans * pre_rotate * pre_scale;

            for (uint32_t i = 0; i < static_cast<uint32_t>(transformed_mesh.Vertices().size()); ++i)
            {
                auto& pos = transformed_mesh.Vertex(i).pos;
                pos.z = -pos.z;
                XMStoreFloat3(&pos, XMVector3TransformCoord(XMLoadFloat3(&pos), adjust_mtx));
                pos.z = -pos.z;
            }

            return transformed_mesh;
        }

    private:
        struct TextureTransferVertexFormat
        {
            XMFLOAT2 ai_tc;
            XMFLOAT2 photo_tc;
        };
        static_assert(sizeof(TextureTransferVertexFormat) == sizeof(float) * 4);

        std::string MeshTexturing(const std::string& mvs_name, const std::string& mesh_name, const std::filesystem::path& working_dir)
        {
            const std::string output_mesh_name = mesh_name + "_Texture";

            const std::string cmd =
                std::format("{} {}.mvs -m {}.glb -o {}.glb --export-type glb --ignore-mask-label 0 --max-texture-size 8192 -w {}",
                    (exe_dir_ / "TextureMesh").string(), mvs_name, mesh_name, output_mesh_name, working_dir.string());
            const int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                throw std::runtime_error(std::format("TextureMesh fails with {}", ret));
            }

            return output_mesh_name;
        }

        void RefillTexture(Mesh& target_mesh, const Mesh& ai_mesh, const Mesh& textured_mesh)
        {
            std::vector<TextureTransferVertexFormat> texture_transfer_vertices =
                this->GenTextureTransferVertices(target_mesh, textured_mesh);

            GpuBuffer vb(gpu_system_, static_cast<uint32_t>(texture_transfer_vertices.size() * sizeof(TextureTransferVertexFormat)),
                D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"vb");
            memcpy(vb.Map(), texture_transfer_vertices.data(), vb.Size());
            vb.Unmap(D3D12_RANGE{0, vb.Size()});

            GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            Texture ai_texture = ai_mesh.AlbedoTexture();
            Ensure4Channel(ai_texture);
            Texture photo_texture = textured_mesh.AlbedoTexture();
            Ensure4Channel(photo_texture);

            GpuTexture2D ai_gpu_tex;
            UploadGpuTexture(gpu_system_, ai_texture, ai_gpu_tex);
            GpuTexture2D photo_gpu_tex;
            UploadGpuTexture(gpu_system_, photo_texture, photo_gpu_tex);

            GpuTexture2D blended_tex = this->BlendTextures(cmd_list, vb, ai_gpu_tex, photo_gpu_tex);
            GpuTexture2D dilated_tmp_tex(gpu_system_, blended_tex.Width(0), blended_tex.Height(0), 1, ColorFmt,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
                L"dilated_tmp_tex");

            this->DilateTexture(cmd_list, blended_tex, dilated_tmp_tex);

            gpu_system_.Execute(std::move(cmd_list));
            gpu_system_.WaitForGpu();

            target_mesh.AlbedoTexture(ReadbackGpuTexture(gpu_system_, blended_tex));

            blended_tex.Reset();
            dilated_tmp_tex.Reset();

            gpu_system_.WaitForGpu();
        }

        std::vector<TextureTransferVertexFormat> GenTextureTransferVertices(Mesh& target_mesh, const Mesh& textured_mesh)
        {
            constexpr float Scale = 1e5f;

            std::set<std::array<int32_t, 3>> unique_int_pos;
            for (uint32_t i = 0; i < target_mesh.Vertices().size(); ++i)
            {
                const auto& pos = target_mesh.Vertex(i).pos;
                std::array<int32_t, 3> int_pos = {static_cast<int32_t>(pos.x * Scale + 0.5f), static_cast<int32_t>(pos.y * Scale + 0.5f),
                    static_cast<int32_t>(pos.z * Scale + 0.5f)};
                unique_int_pos.emplace(std::move(int_pos));
            }

            std::vector<std::array<int32_t, 3>> unique_int_pos_vec(unique_int_pos.begin(), unique_int_pos.end());
            std::vector<uint32_t> vertex_mapping(target_mesh.Vertices().size());
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < static_cast<uint32_t>(target_mesh.Vertices().size()); ++i)
            {
                const auto& pos = target_mesh.Vertex(i).pos;
                const std::array<int32_t, 3> int_pos = {static_cast<int32_t>(pos.x * Scale + 0.5f),
                    static_cast<int32_t>(pos.y * Scale + 0.5f), static_cast<int32_t>(pos.z * Scale + 0.5f)};

                const auto iter = std::lower_bound(unique_int_pos_vec.begin(), unique_int_pos_vec.end(), int_pos);
                assert(*iter == int_pos);

                vertex_mapping[i] = static_cast<uint32_t>(iter - unique_int_pos_vec.begin());
            }

            std::vector<uint32_t> unique_indices(target_mesh.Indices().size());
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < static_cast<uint32_t>(target_mesh.Indices().size()); i += 3)
            {
                for (uint32_t j = 0; j < 3; ++j)
                {
                    unique_indices[i + j] = vertex_mapping[target_mesh.Index(i + j)];
                }
            }

            std::vector<TextureTransferVertexFormat> texture_transfer_vertices;
            texture_transfer_vertices.reserve(textured_mesh.Indices().size());

            for (uint32_t i = 0; i < static_cast<uint32_t>(textured_mesh.Indices().size()); i += 3)
            {
                uint32_t textured_indices[3];
                for (uint32_t j = 0; j < 3; ++j)
                {
                    const auto& pos = textured_mesh.Vertex(textured_mesh.Index(i + j)).pos;
                    const std::array<int32_t, 3> int_pos = {static_cast<int32_t>(pos.x * Scale + 0.5f),
                        static_cast<int32_t>(pos.y * Scale + 0.5f), static_cast<int32_t>(pos.z * Scale + 0.5f)};

                    const auto iter = std::lower_bound(unique_int_pos_vec.begin(), unique_int_pos_vec.end(), int_pos);
                    assert(*iter == int_pos);

                    textured_indices[j] = static_cast<uint32_t>(iter - unique_int_pos_vec.begin());
                }

                bool found = false;
                for (uint32_t j = 0; j < static_cast<uint32_t>(unique_indices.size()); j += 3)
                {
                    // The mesh processed by TextureMesh has a different vertex order and index order than the input one. As a result, we
                    // have to correspondent triangles by checking positions of 3 vertices. If a triangle in ai_mesh has the same vertex
                    // positions as a triangle in textured_mesh, we assume they are correspondent. This assumption is true as long as the
                    // input mesh is from marching cubes.
                    for (uint32_t k = 0; k < 3; ++k)
                    {
                        const uint32_t indices[] = {
                            unique_indices[j + (k + 0) % 3],
                            unique_indices[j + (k + 1) % 3],
                            unique_indices[j + (k + 2) % 3],
                        };

                        if ((textured_indices[0] == indices[0]) && (textured_indices[1] == indices[1]) &&
                            (textured_indices[2] == indices[2]))
                        {
                            for (uint32_t l = 0; l < 3; ++l)
                            {
                                auto& vertex = texture_transfer_vertices.emplace_back();
                                vertex.ai_tc = target_mesh.Vertex(target_mesh.Index(j + (k + l) % 3)).texcoord;
                                vertex.photo_tc = textured_mesh.Vertex(textured_mesh.Index(i + l)).texcoord;
                            }
                            found = true;
                            break;
                        }
                    }

                    if (found)
                    {
                        break;
                    }
                }

                assert(found);
            }

            return texture_transfer_vertices;
        }

        GpuTexture2D BlendTextures(GpuCommandList& cmd_list, GpuBuffer& vb, const GpuTexture2D& ai_tex, const GpuTexture2D& photo_tex)
        {
            auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

            vb.Transition(cmd_list, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

            GpuTexture2D blended_tex(gpu_system_, ai_tex.Width(0), ai_tex.Height(0), 1, ColorFmt,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
                L"blended_tex");
            GpuRenderTargetView rtv(
                gpu_system_, blended_tex, DXGI_FORMAT_UNKNOWN, OffsetHandle(rtv_desc_block_.CpuHandle(), 0, rtv_descriptor_size_));

            GpuShaderResourceView ai_tex_srv(
                gpu_system_, ai_tex, OffsetHandle(refill_texture_srv_uav_desc_block_.CpuHandle(), 0, srv_descriptor_size_));
            GpuShaderResourceView photo_tex_srv(
                gpu_system_, photo_tex, OffsetHandle(refill_texture_srv_uav_desc_block_.CpuHandle(), 1, srv_descriptor_size_));

            D3D12_VERTEX_BUFFER_VIEW vbv{};
            vbv.BufferLocation = vb.GpuVirtualAddress();
            vbv.SizeInBytes = vb.Size();
            vbv.StrideInBytes = sizeof(TextureTransferVertexFormat);
            d3d12_cmd_list->IASetVertexBuffers(0, 1, &vbv);

            d3d12_cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            d3d12_cmd_list->SetPipelineState(refill_texture_pso_.Get());
            d3d12_cmd_list->SetGraphicsRootSignature(refill_texture_root_sig_.Get());

            ID3D12DescriptorHeap* heaps[] = {refill_texture_srv_uav_desc_block_.NativeDescriptorHeap()};
            d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);
            d3d12_cmd_list->SetGraphicsRootDescriptorTable(0, refill_texture_srv_uav_desc_block_.GpuHandle());

            blended_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_RENDER_TARGET);

            D3D12_CPU_DESCRIPTOR_HANDLE rtvs[] = {rtv.CpuHandle()};
            d3d12_cmd_list->OMSetRenderTargets(static_cast<uint32_t>(std::size(rtvs)), rtvs, true, nullptr);

            float clear_clr[] = {0, 0, 0, 0};
            d3d12_cmd_list->ClearRenderTargetView(rtvs[0], clear_clr, 0, nullptr);

            D3D12_VIEWPORT viewport{0, 0, static_cast<float>(blended_tex.Width(0)), static_cast<float>(blended_tex.Height(0)), 0, 1};
            d3d12_cmd_list->RSSetViewports(1, &viewport);

            D3D12_RECT scissor_rc{0, 0, static_cast<LONG>(blended_tex.Width(0)), static_cast<LONG>(blended_tex.Height(0))};
            d3d12_cmd_list->RSSetScissorRects(1, &scissor_rc);

            d3d12_cmd_list->DrawInstanced(static_cast<uint32_t>(vb.Size() / sizeof(TextureTransferVertexFormat)), 1, 0, 0);

            return blended_tex;
        }

        void DilateTexture(GpuCommandList& cmd_list, GpuTexture2D& tex, GpuTexture2D& tmp_tex)
        {
            constexpr uint32_t BlockDim = 16;

            auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

            d3d12_cmd_list->SetPipelineState(dilate_pso_.Get());
            d3d12_cmd_list->SetComputeRootSignature(dilate_root_sig_.Get());

            ID3D12DescriptorHeap* heaps[] = {dilate_srv_uav_desc_block_.NativeDescriptorHeap()};
            d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);

            GpuTexture2D* texs[] = {&tex, &tmp_tex};
            for (uint32_t i = 0; i < DilateTimes; ++i)
            {
                const uint32_t src = i & 1;
                const uint32_t dst = src ? 0 : 1;

                texs[src]->Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
                texs[dst]->Transition(cmd_list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

                GpuShaderResourceView srv(
                    gpu_system_, *texs[src], OffsetHandle(dilate_srv_uav_desc_block_.CpuHandle(), i * 2 + 0, srv_descriptor_size_));
                GpuUnorderedAccessView uav(
                    gpu_system_, *texs[dst], OffsetHandle(dilate_srv_uav_desc_block_.CpuHandle(), i * 2 + 1, srv_descriptor_size_));

                d3d12_cmd_list->SetComputeRootDescriptorTable(
                    0, OffsetHandle(dilate_srv_uav_desc_block_.GpuHandle(), i * 2 + 0, srv_descriptor_size_));
                d3d12_cmd_list->SetComputeRootDescriptorTable(
                    1, OffsetHandle(dilate_srv_uav_desc_block_.GpuHandle(), i * 2 + 1, srv_descriptor_size_));

                d3d12_cmd_list->Dispatch(DivUp(texs[dst]->Width(0), BlockDim), DivUp(texs[dst]->Height(0), BlockDim), 1);
            }

            tmp_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);

            if constexpr (DilateTimes & 1)
            {
                tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COPY_DEST);

                d3d12_cmd_list->CopyResource(tex.NativeTexture(), tmp_tex.NativeTexture());
            }

            tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
        }

    private:
        const std::filesystem::path exe_dir_;

        GpuSystem& gpu_system_;

        GpuDescriptorBlock rtv_desc_block_;
        uint32_t rtv_descriptor_size_;
        uint32_t srv_descriptor_size_;

        ComPtr<ID3D12RootSignature> refill_texture_root_sig_;
        ComPtr<ID3D12PipelineState> refill_texture_pso_;
        GpuDescriptorBlock refill_texture_srv_uav_desc_block_;

        ComPtr<ID3D12RootSignature> dilate_root_sig_;
        ComPtr<ID3D12PipelineState> dilate_pso_;
        GpuDescriptorBlock dilate_srv_uav_desc_block_;

        static constexpr DXGI_FORMAT ColorFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
        static constexpr uint32_t DilateTimes = 4;
    };

    PostProcessor::PostProcessor(const std::filesystem::path& exe_dir, GpuSystem& gpu_system)
        : impl_(std::make_unique<Impl>(exe_dir, gpu_system))
    {
    }

    PostProcessor::~PostProcessor() noexcept = default;

    PostProcessor::PostProcessor(PostProcessor&& other) noexcept = default;
    PostProcessor& PostProcessor::operator=(PostProcessor&& other) noexcept = default;

    Mesh PostProcessor::Process(const MeshReconstruction::Result& recon_input, const Mesh& ai_mesh, const std::filesystem::path& tmp_dir)
    {
        return impl_->Process(recon_input, ai_mesh, tmp_dir);
    }
} // namespace AIHoloImager
