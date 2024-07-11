// Copyright (c) 2024 Minmin Gong
//

#include "PostProcessor.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <set>
#include <span>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"
#include "Gpu/GpuTexture.hpp"

#include "CompiledShader/DilateCs.h"
#include "CompiledShader/RefillTexturePs.h"
#include "CompiledShader/RefillTextureVs.h"

using namespace DirectX;

namespace AIHoloImager
{
    class PostProcessor::Impl
    {
    public:
        Impl(const std::filesystem::path& exe_dir, GpuSystem& gpu_system) : exe_dir_(exe_dir), gpu_system_(gpu_system)
        {
            {
                const ShaderInfo shaders[] = {
                    {RefillTextureVs_shader, 0, 0, 0},
                    {RefillTexturePs_shader, 0, 2, 0},
                };

                const DXGI_FORMAT rtv_formats[] = {ColorFmt};

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::None;
                states.conservative_raster = true;
                states.depth_enable = false;
                states.rtv_formats = rtv_formats;
                states.dsv_format = DXGI_FORMAT_UNKNOWN;

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

                const D3D12_INPUT_ELEMENT_DESC input_elems[] = {
                    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                };

                refill_texture_pipeline_ =
                    GpuRenderPipeline(gpu_system_, shaders, input_elems, std::span(&bilinear_sampler_desc, 1), states);
            }
            {
                const ShaderInfo shader = {DilateCs_shader, 0, 1, 1};
                dilate_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
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

            target_mesh.AlbedoTexture(ReadbackGpuTexture(gpu_system_, blended_tex));
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

            auto rtv_desc_block = gpu_system_.AllocRtvDescBlock(1);
            const uint32_t rtv_descriptor_size = gpu_system_.RtvDescSize();

            GpuTexture2D blended_tex(gpu_system_, ai_tex.Width(0), ai_tex.Height(0), 1, ColorFmt,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
                L"blended_tex");
            GpuRenderTargetView rtv(
                gpu_system_, blended_tex, DXGI_FORMAT_UNKNOWN, OffsetHandle(rtv_desc_block.CpuHandle(), 0, rtv_descriptor_size));

            const float clear_clr[] = {0, 0, 0, 0};
            d3d12_cmd_list->ClearRenderTargetView(rtv.CpuHandle(), clear_clr, 0, nullptr);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&vb, 0, sizeof(TextureTransferVertexFormat)}};
            const uint32_t num_verts = static_cast<uint32_t>(vb.Size() / sizeof(TextureTransferVertexFormat));

            const GpuTexture2D* srv_texs[] = {&ai_tex, &photo_tex};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {{}, {}, {}},
                {{}, srv_texs, {}},
            };

            const GpuCommandList::RenderTargetBinding rt_bindings[] = {{&blended_tex, &rtv}};

            const D3D12_VIEWPORT viewports[] = {
                {0, 0, static_cast<float>(blended_tex.Width(0)), static_cast<float>(blended_tex.Height(0)), 0, 1}};
            const D3D12_RECT scissor_rcs[] = {{0, 0, static_cast<LONG>(blended_tex.Width(0)), static_cast<LONG>(blended_tex.Height(0))}};

            cmd_list.Render(
                refill_texture_pipeline_, vb_bindings, nullptr, num_verts, shader_bindings, rt_bindings, nullptr, viewports, scissor_rcs);

            gpu_system_.DeallocRtvDescBlock(std::move(rtv_desc_block));

            return blended_tex;
        }

        void DilateTexture(GpuCommandList& cmd_list, GpuTexture2D& tex, GpuTexture2D& tmp_tex)
        {
            constexpr uint32_t BlockDim = 16;

            GpuTexture2D* texs[] = {&tex, &tmp_tex};
            for (uint32_t i = 0; i < DilateTimes; ++i)
            {
                const uint32_t src = i & 1;
                const uint32_t dst = src ? 0 : 1;

                const GpuTexture2D* srv_texs[] = {texs[src]};
                GpuTexture2D* uav_texs[] = {texs[dst]};
                const GpuCommandList::ShaderBinding shader_binding = {{}, srv_texs, uav_texs};
                cmd_list.Compute(
                    dilate_pipeline_, DivUp(texs[dst]->Width(0), BlockDim), DivUp(texs[dst]->Height(0), BlockDim), 1, shader_binding);
            }

            if constexpr (DilateTimes & 1)
            {
                tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COPY_DEST);

                auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();
                d3d12_cmd_list->CopyResource(tex.NativeTexture(), tmp_tex.NativeTexture());

                tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
            }
        }

    private:
        const std::filesystem::path exe_dir_;

        GpuSystem& gpu_system_;

        GpuRenderPipeline refill_texture_pipeline_;
        GpuComputePipeline dilate_pipeline_;

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
