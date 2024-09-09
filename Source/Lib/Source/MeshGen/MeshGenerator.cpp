// Copyright (c) 2024 Minmin Gong
//

#include "MeshGenerator.hpp"

#include <array>
#include <cassert>
#include <format>
#include <iostream>
#include <set>

#include <DirectXPackedVector.h>
#include <directx/d3d12.h>
#include <xatlas/xatlas.h>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuTexture.hpp"

#include "CompiledShader/DilateCs.h"
#include "CompiledShader/FlattenPs.h"
#include "CompiledShader/FlattenVs.h"
#include "CompiledShader/GenShadowMapVs.h"
#include "CompiledShader/MergeTextureCs.h"
#include "CompiledShader/ProjectTextureCs.h"
#include "CompiledShader/ResolveTextureCs.h"

using namespace DirectX;
using namespace DirectX::PackedVector;

namespace AIHoloImager
{
    class MeshGenerator::Impl
    {
    public:
        Impl(const std::filesystem::path& exe_dir, GpuSystem& gpu_system, PythonSystem& python_system)
            : exe_dir_(exe_dir), gpu_system_(gpu_system), python_system_(python_system)
        {
            mesh_generator_module_ = python_system_.Import("MeshGenerator");
            mesh_generator_class_ = python_system_.GetAttr(*mesh_generator_module_, "MeshGenerator");
            mesh_generator_ = python_system_.CallObject(*mesh_generator_class_);
            mesh_generator_gen_nerf_method_ = python_system_.GetAttr(*mesh_generator_, "GenNeRF");
            mesh_generator_gen_pos_mesh_method_ = python_system_.GetAttr(*mesh_generator_, "GenPosMesh");
            mesh_generator_query_colors_method_ = python_system_.GetAttr(*mesh_generator_, "QueryColors");

            {
                flatten_cb_ = ConstantBuffer<FlattenConstantBuffer>(gpu_system_, 1, L"flatten_cb_");

                const ShaderInfo shaders[] = {
                    {FlattenVs_shader, 1, 0, 0},
                    {FlattenPs_shader, 0, 0, 0},
                };

                const DXGI_FORMAT rtv_formats[] = {PositionFmt, NormalFmt};

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::None;
                states.conservative_raster = true;
                states.depth_enable = false;
                states.rtv_formats = rtv_formats;
                states.dsv_format = DXGI_FORMAT_UNKNOWN;

                const D3D12_INPUT_ELEMENT_DESC input_elems[] = {
                    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                };

                flatten_pipeline_ = GpuRenderPipeline(gpu_system_, shaders, input_elems, {}, states);
            }
            {
                gen_shadow_map_cb_ = ConstantBuffer<GenShadowMapConstantBuffer>(gpu_system_, 1, L"gen_shadow_map_cb_");

                const ShaderInfo shaders[] = {
                    {GenShadowMapVs_shader, 1, 0, 0},
                    {{}, 0, 0, 0},
                };

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::None;
                states.depth_enable = true;
                states.rtv_formats = {};
                states.dsv_format = DepthFmt;

                const D3D12_INPUT_ELEMENT_DESC input_elems[] = {
                    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                };

                gen_shadow_map_pipeline_ = GpuRenderPipeline(gpu_system_, shaders, input_elems, {}, states);
            }
            {
                project_tex_cb_ = ConstantBuffer<ProjectTextureConstantBuffer>(gpu_system_, 1, L"project_tex_cb_");

                D3D12_STATIC_SAMPLER_DESC sampler_desc[2]{};

                auto& point_sampler_desc = sampler_desc[0];
                point_sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
                point_sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                point_sampler_desc.MaxAnisotropy = 16;
                point_sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                point_sampler_desc.MinLOD = 0.0f;
                point_sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
                point_sampler_desc.ShaderRegister = 0;

                auto& bilinear_sampler_desc = sampler_desc[1];
                bilinear_sampler_desc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                bilinear_sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                bilinear_sampler_desc.MaxAnisotropy = 16;
                bilinear_sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                bilinear_sampler_desc.MinLOD = 0.0f;
                bilinear_sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
                bilinear_sampler_desc.ShaderRegister = 1;

                const ShaderInfo shader = {ProjectTextureCs_shader, 1, 4, 1};
                project_texture_pipeline_ = GpuComputePipeline(gpu_system_, shader, std::span{sampler_desc});
            }
            {
                resolve_texture_cb_ = ConstantBuffer<ResolveTextureConstantBuffer>(gpu_system_, 1, L"resolve_texture_cb_");

                const ShaderInfo shader = {ResolveTextureCs_shader, 1, 2, 4};
                resolve_texture_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                merge_texture_cb_ = ConstantBuffer<MergeTextureConstantBuffer>(gpu_system_, 1, L"merge_texture_cb_");

                const ShaderInfo shader = {MergeTextureCs_shader, 1, 2, 1};
                merge_texture_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                dilate_cb_ = ConstantBuffer<DilateConstantBuffer>(gpu_system_, 1, L"dilate_cb_");

                const ShaderInfo shader = {DilateCs_shader, 1, 1, 1};
                dilate_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        Mesh Generate(std::span<const Texture> input_images, uint32_t texture_size, const StructureFromMotion::Result& sfm_input,
            const MeshReconstruction::Result& recon_input, const std::filesystem::path& tmp_dir)
        {
            assert(input_images.size() == 6);
            assert(input_images[0].Width() == 320);
            assert(input_images[0].Height() == 320);
            assert(input_images[0].NumChannels() == 3);

#ifdef AIHI_KEEP_INTERMEDIATES
            const auto output_dir = tmp_dir / "Texture";
            std::filesystem::create_directories(output_dir);
#endif

            std::cout << "Generating mesh from images...\n";

            const Mesh pos_only_mesh = this->GenMeshFromImages(input_images);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(pos_only_mesh, output_dir / "AiMeshPosOnly.glb");
#endif

            DirectX::BoundingOrientedBox world_obb;
            const XMMATRIX model_mtx = this->CalcModelMatrix(pos_only_mesh, recon_input, world_obb);

            std::cout << "Unwrapping UV...\n";

            std::vector<uint32_t> vertex_mapping;
            Mesh pos_uv_mesh = this->UnwrapUv(pos_only_mesh, texture_size, vertex_mapping);

            std::cout << "Generating texture...\n";

            GpuTexture2D flatten_pos_tex;
            GpuTexture2D flatten_normal_tex;
            this->FlattenMesh(pos_only_mesh, pos_uv_mesh, vertex_mapping, model_mtx, texture_size, flatten_pos_tex, flatten_normal_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                Texture normal_tex(flatten_normal_tex.Width(0), flatten_normal_tex.Height(0), 4);
                flatten_normal_tex.Readback(gpu_system_, cmd_list, 0, normal_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));

                SaveTexture(normal_tex, output_dir / "FlattenNormal.png");
            }
#endif

            GpuReadbackBuffer counter_cpu_buff;
            GpuReadbackBuffer pos_cpu_buff;
            GpuBuffer uv_buff;
            GpuTexture2D color_gpu_tex;
            {
                GpuBuffer counter_buff;
                GpuBuffer pos_buff;
                color_gpu_tex = this->GenTextureFromPhotos(pos_uv_mesh, model_mtx, world_obb, flatten_pos_tex, flatten_normal_tex,
                    sfm_input, texture_size, counter_buff, uv_buff, pos_buff, tmp_dir);

                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

#ifdef AIHI_KEEP_INTERMEDIATES
                {
                    SaveMesh(pos_uv_mesh, output_dir / "AiMeshTextured.glb");

                    Texture projective_tex(texture_size, texture_size, 4);
                    color_gpu_tex.Readback(gpu_system_, cmd_list, 0, projective_tex.Data());
                    SaveTexture(projective_tex, output_dir / "Projective.png");
                }
#endif

                counter_cpu_buff = GpuReadbackBuffer(gpu_system_, counter_buff.Size(), L"counter_cpu_buff");
                cmd_list.Copy(counter_cpu_buff, counter_buff);

                pos_cpu_buff = GpuReadbackBuffer(gpu_system_, pos_buff.Size(), L"pos_cpu_buff");
                cmd_list.Copy(pos_cpu_buff, pos_buff);

                gpu_system_.Execute(std::move(cmd_list));
            }

            {
                const XMVECTOR center = XMLoadFloat3(&recon_input.obb.Center);
                const XMMATRIX pre_trans = XMMatrixTranslationFromVector(-center);
                const XMMATRIX pre_rotate = XMMatrixRotationQuaternion(XMQuaternionInverse(XMLoadFloat4(&recon_input.obb.Orientation))) *
                                            XMMatrixRotationZ(XM_PI / 2) * XMMatrixRotationX(XM_PI);
                const XMMATRIX handedness = XMMatrixScaling(1, 1, -1);

                const XMMATRIX adjust_mtx = model_mtx * pre_trans * pre_rotate * handedness;

                for (uint32_t i = 0; i < static_cast<uint32_t>(pos_uv_mesh.Vertices().size()); ++i)
                {
                    auto& pos = pos_uv_mesh.Vertex(i).pos;
                    XMStoreFloat3(&pos, XMVector3TransformCoord(XMLoadFloat3(&pos), adjust_mtx));
                }
            }
            gpu_system_.WaitForGpu();

            this->MergeTexture(counter_cpu_buff, uv_buff, pos_cpu_buff, color_gpu_tex);

            Texture merged_tex(texture_size, texture_size, 4);
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                GpuTexture2D dilated_tmp_gpu_tex(gpu_system_, color_gpu_tex.Width(0), color_gpu_tex.Height(0), 1, ColorFmt,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, L"dilated_tmp_tex");

                GpuTexture2D* dilated_gpu_tex = this->DilateTexture(cmd_list, color_gpu_tex, dilated_tmp_gpu_tex);

                dilated_gpu_tex->Readback(gpu_system_, cmd_list, 0, merged_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));
            }

            pos_uv_mesh.AlbedoTexture() = std::move(merged_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(pos_uv_mesh, output_dir / "AiMesh.glb");
#endif

            return pos_uv_mesh;
        }

    private:
        Mesh GenMeshFromImages(std::span<const Texture> input_images)
        {
            auto args = python_system_.MakeTuple(1);
            {
                const uint32_t num_images = static_cast<uint32_t>(input_images.size());
                auto imgs_args = python_system_.MakeTuple(num_images);
                for (uint32_t i = 0; i < num_images; ++i)
                {
                    const auto& input_image = input_images[i];
                    auto image = python_system_.MakeObject(
                        std::span<const std::byte>(reinterpret_cast<const std::byte*>(input_image.Data()), input_image.DataSize()));
                    python_system_.SetTupleItem(*imgs_args, i, std::move(image));
                }
                python_system_.SetTupleItem(*args, 0, std::move(imgs_args));
            }

            python_system_.CallObject(*mesh_generator_gen_nerf_method_, *args);

            Mesh pos_only_mesh;
            {
                const auto verts_faces = python_system_.CallObject(*mesh_generator_gen_pos_mesh_method_);

                const auto verts = python_system_.GetTupleItem(*verts_faces, 0);
                const auto faces = python_system_.GetTupleItem(*verts_faces, 1);

                const auto positions = python_system_.ToSpan<const XMFLOAT3>(*verts);
                const auto indices = python_system_.ToSpan<const uint32_t>(*faces);

                pos_only_mesh = this->CleanMesh(positions, indices);
            }

            return pos_only_mesh;
        }

        Mesh CleanMesh(std::span<const XMFLOAT3> positions, std::span<const uint32_t> indices)
        {
            constexpr float Scale = 1e5f;

            std::set<std::array<int32_t, 3>> unique_int_pos;
            for (uint32_t i = 0; i < positions.size(); ++i)
            {
                const XMFLOAT3& pos = positions[i];
                std::array<int32_t, 3> int_pos = {static_cast<int32_t>(pos.x * Scale + 0.5f), static_cast<int32_t>(pos.y * Scale + 0.5f),
                    static_cast<int32_t>(pos.z * Scale + 0.5f)};
                unique_int_pos.emplace(std::move(int_pos));
            }

            Mesh ret_mesh(static_cast<uint32_t>(unique_int_pos.size()), static_cast<uint32_t>(indices.size()));

            std::vector<std::array<int32_t, 3>> unique_int_pos_vec(unique_int_pos.begin(), unique_int_pos.end());
            std::vector<uint32_t> vertex_mapping(positions.size());
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < positions.size(); ++i)
            {
                const XMFLOAT3& pos = positions[i];
                const std::array<int32_t, 3> int_pos = {static_cast<int32_t>(pos.x * Scale + 0.5f),
                    static_cast<int32_t>(pos.y * Scale + 0.5f), static_cast<int32_t>(pos.z * Scale + 0.5f)};

                const auto iter = std::lower_bound(unique_int_pos_vec.begin(), unique_int_pos_vec.end(), int_pos);
                assert(*iter == int_pos);

                vertex_mapping[i] = static_cast<uint32_t>(iter - unique_int_pos_vec.begin());

                auto& pos_only_vert = ret_mesh.Vertex(vertex_mapping[i]);
                pos_only_vert.pos = pos;
                pos_only_vert.texcoord = XMFLOAT2(0, 0);
            }

            uint32_t num_faces = 0;
            for (size_t i = 0; i < indices.size(); i += 3)
            {
                uint32_t face[3];
                for (uint32_t j = 0; j < 3; ++j)
                {
                    face[j] = vertex_mapping[indices[i + j]];
                }

                bool degenerated = false;
                for (uint32_t j = 0; j < 3; ++j)
                {
                    if (face[j] == face[(j + 1) % 3])
                    {
                        degenerated = true;
                        break;
                    }
                }

                if (!degenerated)
                {
                    for (uint32_t j = 0; j < 3; ++j)
                    {
                        ret_mesh.Index(num_faces * 3 + j) = face[j];
                    }
                    ++num_faces;
                }
            }

            ret_mesh.ResizeIndices(num_faces * 3);

            this->RemoveSmallComponents(ret_mesh);
            return ret_mesh;
        }

        void RemoveSmallComponents(Mesh& mesh)
        {
            std::vector<uint32_t> num_neighboring_faces(mesh.Vertices().size(), 0);
            std::vector<uint32_t> neighboring_face_indices(mesh.Indices().size());
            const auto mesh_indices = mesh.Indices();
            for (uint32_t i = 0; i < static_cast<uint32_t>(mesh.Indices().size() / 3); ++i)
            {
                const uint32_t base_index = i * 3;
                for (uint32_t j = 0; j < 3; ++j)
                {
                    bool degenerated = false;
                    for (uint32_t k = 0; k < j; ++k)
                    {
                        if (mesh_indices[base_index + j] == mesh_indices[base_index + k])
                        {
                            degenerated = true;
                            break;
                        }
                    }

                    if (!degenerated)
                    {
                        const uint32_t vi = mesh_indices[base_index + j];
                        neighboring_face_indices[base_index + j] = num_neighboring_faces[vi];
                        ++num_neighboring_faces[vi];
                    }
                }
            }

            std::vector<uint32_t> base_neighboring_faces(mesh.Vertices().size() + 1);
            base_neighboring_faces[0] = 0;
            for (size_t i = 1; i < mesh.Vertices().size() + 1; ++i)
            {
                base_neighboring_faces[i] = base_neighboring_faces[i - 1] + num_neighboring_faces[i - 1];
            }

            std::vector<uint32_t> neighboring_faces(base_neighboring_faces.back());
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < static_cast<uint32_t>(mesh.Indices().size() / 3); ++i)
            {
                const uint32_t base_index = i * 3;
                for (uint32_t j = 0; j < 3; ++j)
                {
                    bool degenerated = false;
                    for (uint32_t k = 0; k < j; ++k)
                    {
                        if (mesh_indices[base_index + j] == mesh_indices[base_index + k])
                        {
                            degenerated = true;
                            break;
                        }
                    }

                    if (!degenerated)
                    {
                        const uint32_t vi = mesh_indices[base_index + j];
                        neighboring_faces[base_neighboring_faces[vi] + neighboring_face_indices[base_index + j]] = i;
                    }
                }
            }

            std::vector<bool> vertex_occupied(mesh.Vertices().size(), false);
            std::vector<uint32_t> largest_comp_face_indices;
            float largest_bb_extent_sq = 0;
            uint32_t num_comps = 0;
            for (;;)
            {
                uint32_t start_vertex = 0;
                while ((start_vertex < vertex_occupied.size()) && vertex_occupied[start_vertex])
                {
                    ++start_vertex;
                }

                if (start_vertex >= vertex_occupied.size())
                {
                    break;
                }

                std::vector<uint32_t> check_vertices(1, start_vertex);
                std::set<uint32_t> check_vertices_tmp;
                std::set<uint32_t> new_component_faces;
                std::set<uint32_t> new_component_vertices;
                while (!check_vertices.empty())
                {
                    for (uint32_t cvi = 0; cvi < check_vertices.size(); ++cvi)
                    {
                        const uint32_t check_vertex = check_vertices[cvi];
                        for (uint32_t fi = base_neighboring_faces[check_vertex]; fi < base_neighboring_faces[check_vertex + 1]; ++fi)
                        {
                            new_component_faces.insert(neighboring_faces[fi]);
                            for (uint32_t j = 0; j < 3; ++j)
                            {
                                const uint32_t nvi = mesh_indices[neighboring_faces[fi] * 3 + j];
                                if (new_component_vertices.find(nvi) == new_component_vertices.end())
                                {
                                    check_vertices_tmp.insert(nvi);
                                    new_component_vertices.insert(nvi);
                                    vertex_occupied[nvi] = true;
                                }
                            }
                        }
                    }

                    check_vertices = std::vector<uint32_t>(check_vertices_tmp.begin(), check_vertices_tmp.end());
                    check_vertices_tmp.clear();
                }

                if (new_component_faces.empty())
                {
                    break;
                }

                XMVECTOR bb_min = XMVectorSplatX(XMVectorSetX(XMVectorZero(), std::numeric_limits<float>::max()));
                XMVECTOR bb_max = XMVectorSplatX(XMVectorSetX(XMVectorZero(), std::numeric_limits<float>::lowest()));
                for (const uint32_t vi : new_component_vertices)
                {
                    const auto& vert = mesh.Vertex(vi);
                    const XMVECTOR pos = XMLoadFloat3(&vert.pos);
                    bb_min = XMVectorMin(bb_min, pos);
                    bb_max = XMVectorMax(bb_max, pos);
                }

                const float bb_extent_sq = XMVectorGetX(XMVector3LengthSq(bb_max - bb_min));
                if (bb_extent_sq > largest_bb_extent_sq)
                {
                    largest_comp_face_indices.assign(new_component_faces.begin(), new_component_faces.end());
                    largest_bb_extent_sq = bb_extent_sq;
                }

                ++num_comps;
            }

            if (num_comps > 1)
            {
                const auto indices = mesh.Indices();
                std::set<uint32_t> comp_vertex_indices;
                for (uint32_t i = 0; i < largest_comp_face_indices.size(); ++i)
                {
                    for (uint32_t j = 0; j < 3; ++j)
                    {
                        comp_vertex_indices.insert(indices[largest_comp_face_indices[i] * 3 + j]);
                    }
                }

                std::vector<uint32_t> vert_mapping(mesh.Vertices().size(), ~0U);
                uint32_t new_index = 0;
                for (const uint32_t vi : comp_vertex_indices)
                {
                    vert_mapping[vi] = new_index;
                    ++new_index;
                }

                Mesh cleaned_mesh(
                    static_cast<uint32_t>(comp_vertex_indices.size()), static_cast<uint32_t>(largest_comp_face_indices.size() * 3));

                const auto vertices = mesh.Vertices();
                new_index = 0;
                for (const uint32_t vi : comp_vertex_indices)
                {
                    cleaned_mesh.Vertex(new_index) = vertices[vi];
                    ++new_index;
                }

                for (uint32_t i = 0; i < largest_comp_face_indices.size(); ++i)
                {
                    for (uint32_t j = 0; j < 3; ++j)
                    {
                        cleaned_mesh.Index(i * 3 + j) = vert_mapping[indices[largest_comp_face_indices[i] * 3 + j]];
                    }
                }

                mesh = std::move(cleaned_mesh);
            }
        }

        Mesh UnwrapUv(const Mesh& input_mesh, uint32_t texture_size, std::vector<uint32_t>& vertex_mapping)
        {
            Mesh ret_mesh;

            xatlas::Atlas* atlas = xatlas::Create();

            xatlas::MeshDecl mesh_decl;
            mesh_decl.vertexCount = static_cast<uint32_t>(input_mesh.Vertices().size());
            mesh_decl.vertexPositionData = input_mesh.Vertices().data();
            mesh_decl.vertexPositionStride = sizeof(input_mesh.Vertices()[0]);
            mesh_decl.indexCount = static_cast<uint32_t>(input_mesh.Indices().size());
            mesh_decl.indexData = input_mesh.Indices().data();
            mesh_decl.indexFormat = xatlas::IndexFormat::UInt32;

            xatlas::AddMeshError error = xatlas::AddMesh(atlas, mesh_decl, 1);
            if (error == xatlas::AddMeshError::Success)
            {
                xatlas::ChartOptions chart_options;

                xatlas::PackOptions pack_options;
                pack_options.padding = 2;
                pack_options.texelsPerUnit = 0;
                pack_options.resolution = texture_size;

                xatlas::Generate(atlas, chart_options, pack_options);

                ret_mesh = Mesh(0, 0);
                for (uint32_t mi = 0; mi < atlas->meshCount; ++mi)
                {
                    const uint32_t base_vertex = static_cast<uint32_t>(ret_mesh.Vertices().size());

                    const xatlas::Mesh& mesh = atlas->meshes[mi];
                    ret_mesh.ResizeVertices(static_cast<uint32_t>(ret_mesh.Vertices().size() + mesh.vertexCount));
                    vertex_mapping.resize(ret_mesh.Vertices().size());
                    for (uint32_t vi = 0; vi < mesh.vertexCount; ++vi)
                    {
                        const auto& vertex = mesh.vertexArray[vi];
                        const auto& pos = input_mesh.Vertex(vertex.xref).pos;
                        const XMFLOAT2 uv(vertex.uv[0] / atlas->width, vertex.uv[1] / atlas->height);
                        ret_mesh.Vertex(base_vertex + vi) = {pos, uv};

                        vertex_mapping[base_vertex + vi] = vertex.xref;
                    }

                    ret_mesh.ResizeIndices(static_cast<uint32_t>(ret_mesh.Indices().size() + mesh.indexCount));
                    for (uint32_t i = 0; i < mesh.indexCount; ++i)
                    {
                        ret_mesh.Index(i) = base_vertex + mesh.indexArray[i];
                    }
                }
            }

            xatlas::Destroy(atlas);

            if (error != xatlas::AddMeshError::Success)
            {
                throw std::runtime_error(std::format("UV unwrapping failed {}", static_cast<uint32_t>(error)));
            }

            return ret_mesh;
        }

        void MergeTexture(const GpuReadbackBuffer& counter_cpu_buff, const GpuBuffer& uv_buff, const GpuReadbackBuffer& pos_cpu_buff,
            GpuTexture2D& color_gpu_tex)
        {
            const uint32_t count = *counter_cpu_buff.MappedData<uint32_t>();
            if (count > 0)
            {
                const XMFLOAT3* pos = pos_cpu_buff.MappedData<XMFLOAT3>();

                auto query_colors_args = python_system_.MakeTuple(2);
                {
                    auto pos_py = python_system_.MakeObject(
                        std::span<const std::byte>(reinterpret_cast<const std::byte*>(pos), count * sizeof(XMFLOAT3)));
                    python_system_.SetTupleItem(*query_colors_args, 0, std::move(pos_py));
                    python_system_.SetTupleItem(*query_colors_args, 1, python_system_.MakeObject(count));
                }

                auto colors_data = python_system_.CallObject(*mesh_generator_query_colors_method_, *query_colors_args);
                const auto colors = python_system_.ToSpan<const uint32_t>(*colors_data);

                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                GpuUnorderedAccessView merged_uav(gpu_system_, color_gpu_tex);

                merge_texture_cb_->buffer_size = count;
                merge_texture_cb_.UploadToGpu();

                GpuShaderResourceView uv_srv(gpu_system_, uv_buff, DXGI_FORMAT_R32_UINT);

                GpuUploadBuffer color_buff(gpu_system_, colors.data(), count * sizeof(uint32_t), L"color_buff");
                GpuShaderResourceView color_srv(gpu_system_, color_buff, DXGI_FORMAT_R8G8B8A8_UNORM);

                constexpr uint32_t BlockDim = 256;

                const GeneralConstantBuffer* cbs[] = {&merge_texture_cb_};
                const GpuShaderResourceView* srvs[] = {&uv_srv, &color_srv};
                GpuUnorderedAccessView* uavs[] = {&merged_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(merge_texture_pipeline_, DivUp(count, BlockDim), 1, 1, shader_binding);

                gpu_system_.Execute(std::move(cmd_list));
                gpu_system_.WaitForGpu();
            }
        }

        XMMATRIX CalcModelMatrix(const Mesh& mesh, const MeshReconstruction::Result& recon_input, DirectX::BoundingOrientedBox& world_obb)
        {
            XMMATRIX model_mtx = XMLoadFloat4x4(&recon_input.transform);
            model_mtx *= XMMatrixScaling(1, 1, -1);    // RH to LH
            std::swap(model_mtx.r[1], model_mtx.r[2]); // Swap Y and Z

            DirectX::BoundingOrientedBox ai_obb;
            BoundingOrientedBox::CreateFromPoints(ai_obb, mesh.Vertices().size(), &mesh.Vertices()[0].pos, sizeof(mesh.Vertices()[0]));

            DirectX::BoundingOrientedBox transformed_ai_obb;
            ai_obb.Transform(transformed_ai_obb, model_mtx);

            const float scale_x = transformed_ai_obb.Extents.x / recon_input.obb.Extents.x;
            const float scale_y = transformed_ai_obb.Extents.y / recon_input.obb.Extents.y;
            const float scale_z = transformed_ai_obb.Extents.z / recon_input.obb.Extents.z;
            const float scale = 1 / std::max({scale_x, scale_y, scale_z});

            model_mtx = XMMatrixScaling(scale, scale, scale) * model_mtx;
            ai_obb.Transform(world_obb, model_mtx);

            return model_mtx * XMMatrixScaling(1, 1, -1);
        }

        void FlattenMesh(const Mesh& pos_only_mesh, const Mesh& pos_uv_mesh, const std::vector<uint32_t>& vertex_mapping,
            const XMMATRIX& model_mtx, uint32_t texture_size, GpuTexture2D& flatten_pos_tex, GpuTexture2D& flatten_normal_tex)
        {
            std::vector<XMFLOAT3> normals(pos_only_mesh.Vertices().size(), XMFLOAT3(0, 0, 0));
            for (uint32_t i = 0; i < static_cast<uint32_t>(pos_only_mesh.Indices().size()); i += 3)
            {
                const uint32_t ind[] = {pos_only_mesh.Index(i + 0), pos_only_mesh.Index(i + 1), pos_only_mesh.Index(i + 2)};
                const XMVECTOR pos[] = {XMLoadFloat3(&pos_only_mesh.Vertex(ind[0]).pos), XMLoadFloat3(&pos_only_mesh.Vertex(ind[1]).pos),
                    XMLoadFloat3(&pos_only_mesh.Vertex(ind[2]).pos)};
                const XMVECTOR edge[] = {pos[1] - pos[0], pos[2] - pos[0]};
                const XMVECTOR face_normal = XMVector3Cross(edge[0], edge[1]);
                for (uint32_t j = 0; j < 3; ++j)
                {
                    XMFLOAT3& normal3 = normals[ind[j]];
                    XMVECTOR normal = XMLoadFloat3(&normal3);
                    normal += face_normal;
                    XMStoreFloat3(&normal3, normal);
                }
            }

            for (size_t i = 0; i < normals.size(); ++i)
            {
                XMFLOAT3& normal3 = normals[i];
                XMVECTOR normal = XMLoadFloat3(&normal3);
                normal = XMVector3Normalize(normal);
                XMStoreFloat3(&normal3, normal);
            }

            struct VertexNormalFormat
            {
                DirectX::XMFLOAT3 pos;
                DirectX::XMFLOAT3 normal;
                DirectX::XMFLOAT2 texcoord;
            };

            std::vector<VertexNormalFormat> vertices(pos_uv_mesh.Vertices().size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(pos_uv_mesh.Vertices().size()); ++i)
            {
                const auto& vertex = pos_uv_mesh.Vertex(i);
                vertices[i].pos = vertex.pos;
                vertices[i].texcoord = vertex.texcoord;

                vertices[i].normal = normals[vertex_mapping[i]];
            }

            GpuBuffer vb(gpu_system_, static_cast<uint32_t>(vertices.size() * sizeof(VertexNormalFormat)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"vb");
            memcpy(vb.Map(), vertices.data(), vb.Size());
            vb.Unmap(D3D12_RANGE{0, vb.Size()});

            GpuBuffer ib(gpu_system_, static_cast<uint32_t>(pos_uv_mesh.Indices().size() * sizeof(uint32_t)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"ib");
            memcpy(ib.Map(), pos_uv_mesh.Indices().data(), ib.Size());
            ib.Unmap(D3D12_RANGE{0, ib.Size()});

            const uint32_t num_indices = static_cast<uint32_t>(pos_uv_mesh.Indices().size());

            flatten_pos_tex = GpuTexture2D(gpu_system_, texture_size, texture_size, 1, PositionFmt, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COMMON, L"flatten_pos_tex");
            flatten_normal_tex = GpuTexture2D(gpu_system_, texture_size, texture_size, 1, NormalFmt,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON, L"flatten_normal_tex");

            GpuRenderTargetView pos_rtv(gpu_system_, flatten_pos_tex);
            GpuRenderTargetView normal_rtv(gpu_system_, flatten_normal_tex);

            GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

            XMStoreFloat4x4(&flatten_cb_->model_mtx, XMMatrixTranspose(model_mtx));
            XMStoreFloat4x4(&flatten_cb_->model_it_mtx, XMMatrixInverse(nullptr, model_mtx));
            flatten_cb_.UploadToGpu();

            flatten_pos_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_RENDER_TARGET);
            flatten_normal_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_RENDER_TARGET);

            const float clear_clr[] = {0, 0, 0, 0};
            d3d12_cmd_list->ClearRenderTargetView(pos_rtv.CpuHandle(), clear_clr, 0, nullptr);
            d3d12_cmd_list->ClearRenderTargetView(normal_rtv.CpuHandle(), clear_clr, 0, nullptr);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&vb, 0, sizeof(VertexNormalFormat)}};
            const GpuCommandList::IndexBufferBinding ib_binding = {&ib, 0, DXGI_FORMAT_R32_UINT};

            const GeneralConstantBuffer* cbs[] = {&flatten_cb_};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {cbs, {}, {}},
                {{}, {}, {}},
            };

            const GpuRenderTargetView* rtvs[] = {&pos_rtv, &normal_rtv};

            const D3D12_VIEWPORT viewports[] = {{0, 0, static_cast<float>(texture_size), static_cast<float>(texture_size), 0, 1}};
            const D3D12_RECT scissor_rcs[] = {{0, 0, static_cast<LONG>(texture_size), static_cast<LONG>(texture_size)}};

            cmd_list.Render(
                flatten_pipeline_, vb_bindings, &ib_binding, num_indices, shader_bindings, rtvs, nullptr, viewports, scissor_rcs);

            flatten_pos_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
            flatten_normal_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);

            gpu_system_.Execute(std::move(cmd_list));
            gpu_system_.WaitForGpu();
        }

        GpuTexture2D GenTextureFromPhotos(const Mesh& pos_uv_mesh, const XMMATRIX& model_mtx, const DirectX::BoundingOrientedBox& world_obb,
            const GpuTexture2D& flatten_pos_tex, const GpuTexture2D& flatten_normal_tex, const StructureFromMotion::Result& sfm_input,
            uint32_t texture_size, GpuBuffer& counter_buff, GpuBuffer& uv_buff, GpuBuffer& pos_buff,
            [[maybe_unused]] const std::filesystem::path& tmp_dir)
        {
            GpuBuffer vb(gpu_system_, static_cast<uint32_t>(pos_uv_mesh.Vertices().size() * sizeof(Mesh::VertexFormat)),
                D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"vb");
            memcpy(vb.Map(), pos_uv_mesh.Vertices().data(), vb.Size());
            vb.Unmap(D3D12_RANGE{0, vb.Size()});

            GpuBuffer ib(gpu_system_, static_cast<uint32_t>(pos_uv_mesh.Indices().size() * sizeof(uint32_t)), D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"ib");
            memcpy(ib.Map(), pos_uv_mesh.Indices().data(), ib.Size());
            ib.Unmap(D3D12_RANGE{0, ib.Size()});

            const uint32_t num_indices = static_cast<uint32_t>(pos_uv_mesh.Indices().size());

            GpuTexture2D accum_color_tex(gpu_system_, texture_size, texture_size, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, L"accum_color_tex");
            GpuUnorderedAccessView accum_color_uav(gpu_system_, accum_color_tex);

            {
                GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
                auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

                accum_color_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

                GpuDescriptorBlock uav_desc_block = gpu_system_.AllocShaderVisibleCbvSrvUavDescBlock(1);
                accum_color_uav.CopyTo(uav_desc_block.CpuHandle());

                ID3D12DescriptorHeap* heaps[] = {uav_desc_block.NativeDescriptorHeap()};
                d3d12_cmd_list->SetDescriptorHeaps(static_cast<uint32_t>(std::size(heaps)), heaps);

                const float black[] = {0, 0, 0, 0};
                d3d12_cmd_list->ClearUnorderedAccessViewFloat(
                    uav_desc_block.GpuHandle(), accum_color_uav.CpuHandle(), accum_color_tex.NativeTexture(), black, 0, nullptr);

                gpu_system_.Execute(std::move(cmd_list));

                gpu_system_.DeallocShaderVisibleCbvSrvUavDescBlock(std::move(uav_desc_block));
            }

            GpuShaderResourceView flatten_pos_srv(gpu_system_, flatten_pos_tex, 0);
            GpuShaderResourceView flatten_normal_srv(gpu_system_, flatten_normal_tex, 0);

            GpuTexture2D shadow_map_tex;
            GpuShaderResourceView shadow_map_srv;
            GpuDepthStencilView shadow_map_dsv;

            GpuTexture2D photo_tex;
            GpuShaderResourceView photo_srv;

            for (size_t i = 0; i < sfm_input.views.size(); ++i)
            {
                std::cout << "Projecting images (" << (i + 1) << " / " << sfm_input.views.size() << ")\r";

                const auto& view = sfm_input.views[i];
                const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                if ((intrinsic.width != shadow_map_tex.Width(0)) || (intrinsic.height != shadow_map_tex.Height(0)))
                {
                    shadow_map_tex = GpuTexture2D(gpu_system_, intrinsic.width, intrinsic.height, 1, DXGI_FORMAT_R32_FLOAT,
                        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_COMMON, L"shadow_map_tex");
                    shadow_map_srv = GpuShaderResourceView(gpu_system_, shadow_map_tex);
                    shadow_map_dsv = GpuDepthStencilView(gpu_system_, shadow_map_tex, DepthFmt);
                }

                if ((view.image_mask.Width() != photo_tex.Width(0)) || (view.image_mask.Height() != photo_tex.Height(0)))
                {
                    photo_tex = GpuTexture2D(gpu_system_, view.image_mask.Width(), view.image_mask.Height(), 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"photo_tex");
                    photo_srv = GpuShaderResourceView(gpu_system_, photo_tex);
                }
                photo_tex.Upload(gpu_system_, cmd_list, 0, view.image_mask.Data());

                const XMVECTOR camera_pos = XMVectorSet(
                    static_cast<float>(view.center.x()), static_cast<float>(view.center.y()), -static_cast<float>(view.center.z()), 1);
                const XMVECTOR camera_up_vec = XMVectorSet(-static_cast<float>(view.rotation(1, 0)),
                    -static_cast<float>(view.rotation(1, 1)), static_cast<float>(view.rotation(1, 2)), 0);
                const XMVECTOR camera_forward_vec = XMVectorSet(static_cast<float>(view.rotation(2, 0)),
                    static_cast<float>(view.rotation(2, 1)), -static_cast<float>(view.rotation(2, 2)), 0);

                const XMMATRIX view_mtx = XMMatrixLookAtLH(camera_pos, camera_pos + camera_forward_vec, camera_up_vec);

                XMFLOAT3 corners[DirectX::BoundingOrientedBox::CORNER_COUNT];
                world_obb.GetCorners(corners);

                const XMVECTOR z_col = XMVectorSet(
                    XMVectorGetZ(view_mtx.r[0]), XMVectorGetZ(view_mtx.r[1]), XMVectorGetZ(view_mtx.r[2]), XMVectorGetZ(view_mtx.r[3]));

                float min_z_es = 1e10f;
                float max_z_es = -1e10f;
                for (const auto& corner : corners)
                {
                    XMVECTOR pos = XMVectorSet(corner.x, corner.y, -corner.z, 1);
                    pos = XMVector4Dot(pos, z_col);
                    const float z = XMVectorGetZ(pos);
                    min_z_es = std::min(min_z_es, z);
                    max_z_es = std::max(max_z_es, z);
                }

                const float center_es_z = (max_z_es + min_z_es) / 2;
                const float extent_es_z = (max_z_es - min_z_es) / 2 * 1.05f;

                const float near_plane = center_es_z - extent_es_z;
                const float far_plane = center_es_z + extent_es_z;

                const double fy = intrinsic.k(1, 1);
                const float fov = static_cast<float>(2 * std::atan(intrinsic.height / (2 * fy)));
                const XMMATRIX proj_mtx =
                    XMMatrixPerspectiveFovLH(fov, static_cast<float>(intrinsic.width) / intrinsic.height, near_plane, far_plane);

                XMStoreFloat4x4(&gen_shadow_map_cb_->mvp, XMMatrixTranspose(model_mtx * view_mtx * proj_mtx));
                gen_shadow_map_cb_.UploadToGpu();

                const XMFLOAT2 offset = {
                    static_cast<float>(intrinsic.k(0, 2)) - intrinsic.width / 2,
                    static_cast<float>(intrinsic.k(1, 2)) - intrinsic.height / 2,
                };

                shadow_map_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_DEPTH_WRITE);

                this->GenShadowMap(cmd_list, vb, ib, num_indices, offset, intrinsic, shadow_map_dsv);

                shadow_map_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);
                accum_color_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

                this->ProjectTexture(cmd_list, texture_size, view_mtx, proj_mtx, offset, intrinsic, flatten_pos_srv, flatten_normal_srv,
                    photo_srv, shadow_map_srv, accum_color_uav);

                accum_color_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_COMMON);

#ifdef AIHI_KEEP_INTERMEDIATES
                {
                    Texture color_tex(accum_color_tex.Width(0), accum_color_tex.Height(0), 4);
                    accum_color_tex.Readback(gpu_system_, cmd_list, 0, color_tex.Data());
                    SaveTexture(color_tex, tmp_dir / "Texture" / std::format("Projective_{}.png", i));
                }
#endif

                gpu_system_.Execute(std::move(cmd_list));
                gpu_system_.WaitForGpu();
            }
            std::cout << "\n";

            return this->ResolveTexture(model_mtx, texture_size, accum_color_tex, flatten_pos_srv, counter_buff, uv_buff, pos_buff);
        }

        void GenShadowMap(GpuCommandList& cmd_list, const GpuBuffer& vb, const GpuBuffer& ib, uint32_t num_indices, const XMFLOAT2& offset,
            const StructureFromMotion::PinholeIntrinsic& intrinsic, GpuDepthStencilView& shadow_map_dsv)
        {
            auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();
            d3d12_cmd_list->ClearDepthStencilView(shadow_map_dsv.CpuHandle(), D3D12_CLEAR_FLAG_DEPTH, 1, 0, 0, nullptr);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&vb, 0, sizeof(Mesh::VertexFormat)}};
            const GpuCommandList::IndexBufferBinding ib_binding = {&ib, 0, DXGI_FORMAT_R32_UINT};

            const GeneralConstantBuffer* cbs[] = {&gen_shadow_map_cb_};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {cbs, {}, {}},
                {{}, {}, {}},
            };

            const D3D12_VIEWPORT viewports[] = {
                {offset.x, offset.y, static_cast<float>(intrinsic.width), static_cast<float>(intrinsic.height), 0, 1}};
            const D3D12_RECT scissor_rcs[] = {{0, 0, static_cast<LONG>(intrinsic.width), static_cast<LONG>(intrinsic.height)}};

            cmd_list.Render(gen_shadow_map_pipeline_, vb_bindings, &ib_binding, num_indices, shader_bindings, {}, &shadow_map_dsv,
                viewports, scissor_rcs);
        }

        void ProjectTexture(GpuCommandList& cmd_list, uint32_t texture_size, const XMMATRIX& view_mtx, const XMMATRIX& proj_mtx,
            const XMFLOAT2& offset, const StructureFromMotion::PinholeIntrinsic& intrinsic, const GpuShaderResourceView& flatten_pos_srv,
            const GpuShaderResourceView& flatten_normal_srv, const GpuShaderResourceView& projective_map_srv,
            const GpuShaderResourceView& shadow_map_srv, GpuUnorderedAccessView& accum_color_uav)
        {
            constexpr uint32_t BlockDim = 16;

            XMStoreFloat4x4(&project_tex_cb_->camera_view_proj, XMMatrixTranspose(view_mtx * proj_mtx));
            XMStoreFloat4x4(&project_tex_cb_->camera_view, XMMatrixTranspose(view_mtx));
            XMStoreFloat4x4(&project_tex_cb_->camera_view_it, XMMatrixInverse(nullptr, view_mtx));
            project_tex_cb_->offset = XMFLOAT2(offset.x / intrinsic.width, offset.y / intrinsic.height);
            project_tex_cb_->texture_size = texture_size;
            project_tex_cb_.UploadToGpu();

            const GeneralConstantBuffer* cbs[] = {&project_tex_cb_};
            const GpuShaderResourceView* srvs[] = {&flatten_pos_srv, &flatten_normal_srv, &projective_map_srv, &shadow_map_srv};
            GpuUnorderedAccessView* uavs[] = {&accum_color_uav};

            const GpuCommandList::ShaderBinding cs_shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(project_texture_pipeline_, DivUp(texture_size, BlockDim), DivUp(texture_size, BlockDim), 1, cs_shader_binding);
        }

        GpuTexture2D ResolveTexture(const XMMATRIX& model_mtx, uint32_t texture_size, const GpuTexture2D& accum_color_tex,
            const GpuShaderResourceView& flatten_pos_srv, GpuBuffer& counter_buff, GpuBuffer& uv_buff, GpuBuffer& pos_buff)
        {
            constexpr uint32_t BlockDim = 16;

            XMStoreFloat4x4(&resolve_texture_cb_->inv_model, XMMatrixTranspose(XMMatrixInverse(nullptr, model_mtx)));
            resolve_texture_cb_->texture_size = texture_size;
            resolve_texture_cb_.UploadToGpu();

            GpuCommandList cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            GpuTexture2D color_tex(gpu_system_, texture_size, texture_size, 1, ColorFmt, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, L"color_tex");
            GpuUnorderedAccessView color_uav(gpu_system_, color_tex);

            counter_buff = GpuBuffer(gpu_system_, sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, L"counter_buff");
            {
                GpuUploadBuffer counter_upload_buff(gpu_system_, counter_buff.Size(), L"counter_upload_buff");
                *reinterpret_cast<uint32_t*>(counter_upload_buff.MappedData()) = 0;
                cmd_list.Copy(counter_buff, counter_upload_buff);
            }
            GpuUnorderedAccessView counter_uav(gpu_system_, counter_buff, sizeof(uint32_t));

            const uint32_t max_pos_size = texture_size * texture_size;
            uv_buff = GpuBuffer(gpu_system_, max_pos_size * sizeof(XMUSHORT2), D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, L"uv_buff");
            GpuUnorderedAccessView uv_uav(gpu_system_, uv_buff, sizeof(XMUSHORT2));
            pos_buff = GpuBuffer(gpu_system_, max_pos_size * sizeof(XMFLOAT3), D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, L"pos_buff");
            GpuUnorderedAccessView pos_uav(gpu_system_, pos_buff, sizeof(float));

            GpuShaderResourceView accum_color_srv(gpu_system_, accum_color_tex);

            const GeneralConstantBuffer* cbs[] = {&resolve_texture_cb_};
            const GpuShaderResourceView* srvs[] = {&accum_color_srv, &flatten_pos_srv};
            GpuUnorderedAccessView* uavs[] = {&color_uav, &counter_uav, &uv_uav, &pos_uav};

            const GpuCommandList::ShaderBinding cs_shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(resolve_texture_pipeline_, DivUp(texture_size, BlockDim), DivUp(texture_size, BlockDim), 1, cs_shader_binding);

            gpu_system_.Execute(std::move(cmd_list));

            return color_tex;
        }

        GpuTexture2D* DilateTexture(GpuCommandList& cmd_list, GpuTexture2D& tex, GpuTexture2D& tmp_tex)
        {
            constexpr uint32_t BlockDim = 16;

            dilate_cb_->texture_size = tex.Width(0);
            dilate_cb_.UploadToGpu();

            GpuShaderResourceView tex_srv(gpu_system_, tex);
            GpuShaderResourceView tmp_tex_srv(gpu_system_, tmp_tex);
            GpuUnorderedAccessView tex_uav(gpu_system_, tex);
            GpuUnorderedAccessView tmp_tex_uav(gpu_system_, tmp_tex);

            GpuTexture2D* texs[] = {&tex, &tmp_tex};
            GpuShaderResourceView* tex_srvs[] = {&tex_srv, &tmp_tex_srv};
            GpuUnorderedAccessView* tex_uavs[] = {&tex_uav, &tmp_tex_uav};
            for (uint32_t i = 0; i < DilateTimes; ++i)
            {
                const uint32_t src = i & 1;
                const uint32_t dst = src ? 0 : 1;

                const GeneralConstantBuffer* cbs[] = {&dilate_cb_};
                const GpuShaderResourceView* srvs[] = {tex_srvs[src]};
                GpuUnorderedAccessView* uavs[] = {tex_uavs[dst]};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(
                    dilate_pipeline_, DivUp(texs[dst]->Width(0), BlockDim), DivUp(texs[dst]->Height(0), BlockDim), 1, shader_binding);
            }

            if constexpr (DilateTimes & 1)
            {
                return &tmp_tex;
            }
            else
            {
                return &tex;
            }
        }

    private:
        const std::filesystem::path exe_dir_;

        GpuSystem& gpu_system_;
        PythonSystem& python_system_;

        PyObjectPtr mesh_generator_module_;
        PyObjectPtr mesh_generator_class_;
        PyObjectPtr mesh_generator_;
        PyObjectPtr mesh_generator_gen_nerf_method_;
        PyObjectPtr mesh_generator_gen_pos_mesh_method_;
        PyObjectPtr mesh_generator_query_colors_method_;

        struct FlattenConstantBuffer
        {
            DirectX::XMFLOAT4X4 model_mtx;
            DirectX::XMFLOAT4X4 model_it_mtx;
        };
        ConstantBuffer<FlattenConstantBuffer> flatten_cb_;
        GpuRenderPipeline flatten_pipeline_;

        struct GenShadowMapConstantBuffer
        {
            DirectX::XMFLOAT4X4 mvp;
        };
        ConstantBuffer<GenShadowMapConstantBuffer> gen_shadow_map_cb_;
        GpuRenderPipeline gen_shadow_map_pipeline_;

        struct ProjectTextureConstantBuffer
        {
            DirectX::XMFLOAT4X4 camera_view_proj;
            DirectX::XMFLOAT4X4 camera_view;
            DirectX::XMFLOAT4X4 camera_view_it;
            DirectX::XMFLOAT2 offset;
            uint32_t texture_size;
            uint32_t padding;
        };
        ConstantBuffer<ProjectTextureConstantBuffer> project_tex_cb_;
        GpuComputePipeline project_texture_pipeline_;

        struct ResolveTextureConstantBuffer
        {
            DirectX::XMFLOAT4X4 inv_model;
            uint32_t texture_size;
            uint32_t padding[3];
        };
        ConstantBuffer<ResolveTextureConstantBuffer> resolve_texture_cb_;
        GpuComputePipeline resolve_texture_pipeline_;

        struct MergeTextureConstantBuffer
        {
            uint32_t buffer_size;
            uint32_t padding[3];
        };
        ConstantBuffer<MergeTextureConstantBuffer> merge_texture_cb_;
        GpuComputePipeline merge_texture_pipeline_;

        struct DilateConstantBuffer
        {
            uint32_t texture_size;
            uint32_t padding[3];
        };
        ConstantBuffer<DilateConstantBuffer> dilate_cb_;
        GpuComputePipeline dilate_pipeline_;

        static constexpr DXGI_FORMAT ColorFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
        static constexpr DXGI_FORMAT PositionFmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
        static constexpr DXGI_FORMAT NormalFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
        static constexpr DXGI_FORMAT DepthFmt = DXGI_FORMAT_D32_FLOAT;
        static constexpr uint32_t DilateTimes = 4;
    };

    MeshGenerator::MeshGenerator(const std::filesystem::path& exe_dir, GpuSystem& gpu_system, PythonSystem& python_system)
        : impl_(std::make_unique<Impl>(exe_dir, gpu_system, python_system))
    {
    }

    MeshGenerator::~MeshGenerator() noexcept = default;

    MeshGenerator::MeshGenerator(MeshGenerator&& other) noexcept = default;
    MeshGenerator& MeshGenerator::operator=(MeshGenerator&& other) noexcept = default;

    Mesh MeshGenerator::Generate(std::span<const Texture> input_images, uint32_t texture_size, const StructureFromMotion::Result& sfm_input,
        const MeshReconstruction::Result& recon_input, const std::filesystem::path& tmp_dir)
    {
        return impl_->Generate(std::move(input_images), texture_size, sfm_input, recon_input, tmp_dir);
    }
} // namespace AIHoloImager
