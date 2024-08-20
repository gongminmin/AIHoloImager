// Copyright (c) 2024 Minmin Gong
//

#include "MeshGenerator.hpp"

#include <array>
#include <cassert>
#include <format>
#include <set>

#include <directx/d3d12.h>
#include <xatlas/xatlas.h>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuTexture.hpp"

#include "CompiledShader/DilateCs.h"
#include "CompiledShader/GetPosListCs.h"
#include "CompiledShader/TransferTexturePs.h"
#include "CompiledShader/TransferTextureVs.h"

using namespace DirectX;

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
            mesh_generator_gen_pos_mesh_method_ = python_system_.GetAttr(*mesh_generator_, "GenPosMesh");
            mesh_generator_query_colors_method_ = python_system_.GetAttr(*mesh_generator_, "QueryColors");

            pil_module_ = python_system_.Import("PIL");
            image_class_ = python_system_.GetAttr(*pil_module_, "Image");
            image_frombuffer_method_ = python_system_.GetAttr(*image_class_, "frombuffer");

            {
                const ShaderInfo shaders[] = {
                    {TransferTextureVs_shader, 0, 0, 0},
                    {TransferTexturePs_shader, 0, 1, 0},
                };

                const DXGI_FORMAT rtv_formats[] = {ColorFmt, PositionFmt};

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
                    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                    {"TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, 20, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
                };

                transfer_texture_pipeline_ =
                    GpuRenderPipeline(gpu_system_, shaders, input_elems, std::span(&bilinear_sampler_desc, 1), states);
            }
            {
                const ShaderInfo shader = {GetPosListCs_shader, 0, 1, 3};
                get_pos_list_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {DilateCs_shader, 0, 1, 1};
                dilate_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        Mesh Generate(std::span<const Texture> input_images, uint32_t texture_size, const MeshReconstruction::Result& recon_input,
            const std::filesystem::path& tmp_dir)
        {
            assert(input_images.size() == 6);
            assert(input_images[0].Width() == 320);
            assert(input_images[0].Height() == 320);
            assert(input_images[0].NumChannels() == 3);

#ifdef AIHI_KEEP_INTERMEDIATES
            auto output_dir = tmp_dir / "Mesh";
            std::filesystem::create_directories(output_dir);
#endif

            PyObjectPtr py_input_images[6];
            for (size_t i = 0; i < input_images.size(); ++i)
            {
                auto& input_image = input_images[i];

                auto args = python_system_.MakeTuple(3);
                {
                    python_system_.SetTupleItem(*args, 0, python_system_.MakeObject(L"RGB"));
                }
                {
                    auto size = python_system_.MakeTuple(2);
                    {
                        python_system_.SetTupleItem(*size, 0, python_system_.MakeObject(input_image.Width()));
                        python_system_.SetTupleItem(*size, 1, python_system_.MakeObject(input_image.Height()));
                    }
                    python_system_.SetTupleItem(*args, 1, std::move(size));
                }
                {
                    auto image = python_system_.MakeObject(
                        std::span<const std::byte>(reinterpret_cast<const std::byte*>(input_image.Data()), input_image.DataSize()));
                    python_system_.SetTupleItem(*args, 2, std::move(image));
                }

                py_input_images[i] = python_system_.CallObject(*image_frombuffer_method_, *args);
            }

            auto args = python_system_.MakeTuple(1);
            {
                auto imgs_args = python_system_.MakeTuple(std::size(py_input_images));
                for (uint32_t i = 0; i < std::size(py_input_images); ++i)
                {
                    python_system_.SetTupleItem(*imgs_args, i, std::move(py_input_images[i]));
                }
                python_system_.SetTupleItem(*args, 0, std::move(imgs_args));
            }

            auto verts_faces = python_system_.CallObject(*mesh_generator_gen_pos_mesh_method_, *args);

            Mesh pos_only_mesh;
            {
                auto verts = python_system_.GetTupleItem(*verts_faces, 0);
                auto faces = python_system_.GetTupleItem(*verts_faces, 1);

                auto verts_tobytes_method = python_system_.GetAttr(*verts, "tobytes");
                auto verts_data = python_system_.CallObject(*verts_tobytes_method);
                const auto positions = python_system_.ToSpan<const XMFLOAT3>(*verts_data);

                auto faces_tobytes_method = python_system_.GetAttr(*faces, "tobytes");
                auto faces_data = python_system_.CallObject(*faces_tobytes_method);
                const auto indices = python_system_.ToSpan<const uint32_t>(*faces_data);

                pos_only_mesh = this->CleanMesh(positions, indices);

#ifdef AIHI_KEEP_INTERMEDIATES
                SaveMesh(pos_only_mesh, output_dir / "AiMeshPosOnly.glb");
#endif
            }

            Mesh pos_uv_mesh = this->UnwrapUv(pos_only_mesh, texture_size);

            GpuTexture2D pos_gpu_tex;
            Mesh textured_mesh = this->GenTextureFromPhotos(pos_only_mesh, pos_uv_mesh, recon_input, texture_size, pos_gpu_tex, tmp_dir);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(textured_mesh, output_dir / "AiMeshTextured.glb");
#endif

            GpuReadbackBuffer counter_cpu_buff;
            GpuReadbackBuffer uv_cpu_buff;
            GpuReadbackBuffer pos_cpu_buff;
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                GpuBuffer counter_buff;
                GpuBuffer uv_buff;
                GpuBuffer pos_buff;
                this->PosTexToList(cmd_list, pos_gpu_tex, counter_buff, uv_buff, pos_buff);

                counter_cpu_buff = GpuReadbackBuffer(gpu_system_, counter_buff.Size(), L"counter_cpu_buff");
                cmd_list.Copy(counter_cpu_buff, counter_buff);

                uv_cpu_buff = GpuReadbackBuffer(gpu_system_, uv_buff.Size(), L"uv_cpu_buff");
                cmd_list.Copy(uv_cpu_buff, uv_buff);

                pos_cpu_buff = GpuReadbackBuffer(gpu_system_, pos_buff.Size(), L"pos_cpu_buff");
                cmd_list.Copy(pos_cpu_buff, pos_buff);

                gpu_system_.Execute(std::move(cmd_list));
                gpu_system_.WaitForGpu();
            }

            Texture& blended_tex = textured_mesh.AlbedoTexture();
            this->GenTexture(counter_cpu_buff, uv_cpu_buff, pos_cpu_buff, blended_tex);

            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                GpuTexture2D blended_gpu_tex(gpu_system_, blended_tex.Width(), blended_tex.Height(), 1, ColorFmt,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
                blended_gpu_tex.Upload(gpu_system_, cmd_list, 0, blended_tex.Data());

                GpuTexture2D dilated_tmp_gpu_tex(gpu_system_, blended_tex.Width(), blended_tex.Height(), 1, ColorFmt,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, L"dilated_tmp_tex");

                GpuTexture2D* dilated_gpu_tex = this->DilateTexture(cmd_list, blended_gpu_tex, dilated_tmp_gpu_tex);

                dilated_gpu_tex->Readback(gpu_system_, cmd_list, 0, blended_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));
            }

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(textured_mesh, output_dir / "AiMesh.glb");
#endif

            return textured_mesh;
        }

    private:
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

        Mesh UnwrapUv(const Mesh& input_mesh, uint32_t texture_size)
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
                    for (uint32_t vi = 0; vi < mesh.vertexCount; ++vi)
                    {
                        const auto& vertex = mesh.vertexArray[vi];
                        const auto& pos = input_mesh.Vertex(vertex.xref).pos;
                        const XMFLOAT2 uv(vertex.uv[0] / atlas->width, vertex.uv[1] / atlas->height);
                        ret_mesh.Vertex(base_vertex + vi) = {pos, uv};
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

        void PosTexToList(
            GpuCommandList& cmd_list, const GpuTexture2D& pos_tex, GpuBuffer& counter_buff, GpuBuffer& uv_buff, GpuBuffer& pos_buff)
        {
            constexpr uint32_t BlockDim = 16;

            GpuShaderResourceView pos_tex_srv(gpu_system_, pos_tex);

            counter_buff = GpuBuffer(gpu_system_, sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, L"counter_buff");
            GpuUnorderedAccessView counter_uav(gpu_system_, counter_buff, sizeof(uint32_t));

            const uint32_t max_pos_size = pos_tex.Width(0) * pos_tex.Height(0);
            uv_buff = GpuBuffer(gpu_system_, max_pos_size * sizeof(XMUINT2), D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, L"uv_buff");
            GpuUnorderedAccessView uv_uav(gpu_system_, uv_buff, sizeof(XMUINT2));
            pos_buff = GpuBuffer(gpu_system_, max_pos_size * sizeof(XMFLOAT3), D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, L"pos_buff");
            GpuUnorderedAccessView pos_uav(gpu_system_, pos_buff, sizeof(float));

            const GpuShaderResourceView* srvs[] = {&pos_tex_srv};
            GpuUnorderedAccessView* uavs[] = {&counter_uav, &uv_uav, &pos_uav};
            const GpuCommandList::ShaderBinding shader_binding = {{}, srvs, uavs};
            cmd_list.Compute(
                get_pos_list_pipeline_, DivUp(pos_tex.Width(0), BlockDim), DivUp(pos_tex.Height(0), BlockDim), 1, shader_binding);
        }

        void GenTexture(const GpuReadbackBuffer& counter_cpu_buff, const GpuReadbackBuffer& uv_cpu_buff,
            const GpuReadbackBuffer& pos_cpu_buff, Texture& texture)
        {
            const uint32_t count = *counter_cpu_buff.MappedData<uint32_t>();

            const XMUINT2* uv = uv_cpu_buff.MappedData<XMUINT2>();
            const XMFLOAT3* pos = pos_cpu_buff.MappedData<XMFLOAT3>();

            auto query_colors_args = python_system_.MakeTuple(2);
            {
                auto pos_py = python_system_.MakeObject(
                    std::span<const std::byte>(reinterpret_cast<const std::byte*>(pos), count * sizeof(XMFLOAT3)));
                python_system_.SetTupleItem(*query_colors_args, 0, std::move(pos_py));
                python_system_.SetTupleItem(*query_colors_args, 1, python_system_.MakeObject(count));
            }

            auto colors_py = python_system_.CallObject(*mesh_generator_query_colors_method_, *query_colors_args);

            auto colors_tobytes_method = python_system_.GetAttr(*colors_py, "tobytes");
            auto colors_data = python_system_.CallObject(*colors_tobytes_method);
            const auto colors = python_system_.ToSpan<const uint32_t>(*colors_data);

            uint32_t* tex_data = reinterpret_cast<uint32_t*>(texture.Data());
            const uint32_t texture_size = texture.Width();
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < count; ++i)
            {
                tex_data[uv[i].y * texture_size + uv[i].x] = colors[i];
            }
        }

        Mesh GenTextureFromPhotos(const Mesh& pos_only_mesh, const Mesh& pos_uv_mesh, const MeshReconstruction::Result& recon_input,
            uint32_t texture_size, GpuTexture2D& pos_gpu_tex, const std::filesystem::path& tmp_dir)
        {
            XMMATRIX transform_mtx = XMLoadFloat4x4(&recon_input.transform);
            transform_mtx *= XMMatrixScaling(1, 1, -1);        // RH to LH
            std::swap(transform_mtx.r[1], transform_mtx.r[2]); // Swap Y and Z

            DirectX::BoundingOrientedBox ai_obb;
            BoundingOrientedBox::CreateFromPoints(
                ai_obb, pos_only_mesh.Vertices().size(), &pos_only_mesh.Vertices()[0].pos, sizeof(pos_only_mesh.Vertices()[0]));

            ai_obb.Transform(ai_obb, transform_mtx);

            const float scale_x = ai_obb.Extents.x / recon_input.obb.Extents.x;
            const float scale_y = ai_obb.Extents.y / recon_input.obb.Extents.y;
            const float scale_z = ai_obb.Extents.z / recon_input.obb.Extents.z;
            const float scale = 1 / std::max({scale_x, scale_y, scale_z});

            Mesh transformed_mesh(
                static_cast<uint32_t>(pos_uv_mesh.Vertices().size()), static_cast<uint32_t>(pos_uv_mesh.Indices().size()));

            transform_mtx = XMMatrixScaling(scale, scale, scale) * transform_mtx;
            for (uint32_t i = 0; i < static_cast<uint32_t>(transformed_mesh.Vertices().size()); ++i)
            {
                auto& vertex = transformed_mesh.Vertex(i);

                const auto& pos = pos_uv_mesh.Vertex(i).pos;
                XMStoreFloat3(&vertex.pos, XMVector3TransformCoord(XMLoadFloat3(&pos), transform_mtx));

                vertex.texcoord = XMFLOAT2(0, 0); // TextureMesh can't handle mesh with texture coordinate. Clear it.
            }

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < static_cast<uint32_t>(transformed_mesh.Indices().size()); ++i)
            {
                transformed_mesh.Index(i) = pos_uv_mesh.Index(i);
            }

            const auto working_dir = tmp_dir / "Mvs";
            std::filesystem::create_directories(working_dir);

            const std::string mesh_name = "Temp_Ai";
            SaveMesh(transformed_mesh, working_dir / (mesh_name + ".glb"));

            const std::string output_mesh_name = mesh_name + "_Texture";

            const std::string cmd = std::format("{} Temp.mvs -m {}.glb -o {}.glb --export-type glb --ignore-mask-label 0 "
                                                "--max-texture-size 8192 --process-priority 0 -w {}",
                (exe_dir_ / "TextureMesh").string(), mesh_name, output_mesh_name, working_dir.string());
            const int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                throw std::runtime_error(std::format("TextureMesh fails with {}", ret));
            }

            Mesh textured_mesh = LoadMesh(working_dir / (output_mesh_name + ".glb"));

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < static_cast<uint32_t>(transformed_mesh.Vertices().size()); ++i)
            {
                transformed_mesh.Vertex(i).texcoord = pos_uv_mesh.Vertex(i).texcoord;
            }
            this->RefillTexture(transformed_mesh, pos_uv_mesh, textured_mesh, texture_size, pos_gpu_tex);

            const XMVECTOR center = XMLoadFloat3(&recon_input.obb.Center);
            const XMMATRIX pre_trans = XMMatrixTranslationFromVector(-center);
            const XMMATRIX pre_rotate = XMMatrixRotationQuaternion(XMQuaternionInverse(XMLoadFloat4(&recon_input.obb.Orientation))) *
                                        XMMatrixRotationZ(XM_PI / 2) * XMMatrixRotationX(XM_PI);
            const XMMATRIX handedness = XMMatrixScaling(1, 1, -1);

            const XMMATRIX adjust_mtx = handedness * pre_trans * pre_rotate * handedness;

            for (uint32_t i = 0; i < static_cast<uint32_t>(transformed_mesh.Vertices().size()); ++i)
            {
                auto& pos = transformed_mesh.Vertex(i).pos;
                XMStoreFloat3(&pos, XMVector3TransformCoord(XMLoadFloat3(&pos), adjust_mtx));
            }

            return transformed_mesh;
        }

        struct TextureTransferVertexFormat
        {
            XMFLOAT3 pos;
            XMFLOAT2 ai_tc;
            XMFLOAT2 photo_tc;
        };
        static_assert(sizeof(TextureTransferVertexFormat) == sizeof(float) * 7);

        void RefillTexture(
            Mesh& target_mesh, const Mesh& pos_uv_mesh, const Mesh& textured_mesh, uint32_t texture_size, GpuTexture2D& pos_gpu_tex)
        {
            GpuBuffer vb(gpu_system_, static_cast<uint32_t>(textured_mesh.Indices().size() * sizeof(TextureTransferVertexFormat)),
                D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, L"vb");
            this->GenTextureTransferVertices(
                target_mesh, pos_uv_mesh, textured_mesh, reinterpret_cast<TextureTransferVertexFormat*>(vb.Map()));
            vb.Unmap(D3D12_RANGE{0, vb.Size()});

            Texture photo_texture = textured_mesh.AlbedoTexture();
            Ensure4Channel(photo_texture);

            auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            GpuTexture2D photo_gpu_tex(gpu_system_, photo_texture.Width(), photo_texture.Height(), 1, ColorFmt, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON);
            photo_gpu_tex.Upload(gpu_system_, cmd_list, 0, photo_texture.Data());

            GpuTexture2D color_gpu_tex;
            this->TransferTexture(cmd_list, vb, photo_gpu_tex, texture_size, color_gpu_tex, pos_gpu_tex);

            Texture color_texture(color_gpu_tex.Width(0), color_gpu_tex.Height(0), FormatSize(color_gpu_tex.Format()));
            color_gpu_tex.Readback(gpu_system_, cmd_list, 0, color_texture.Data());
            gpu_system_.Execute(std::move(cmd_list));

            target_mesh.AlbedoTexture() = std::move(color_texture);
        }

        void GenTextureTransferVertices(const Mesh& target_mesh, const Mesh& pos_uv_mesh, const Mesh& textured_mesh,
            TextureTransferVertexFormat* texture_transfer_vertices)
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
            for (uint32_t i = 0; i < static_cast<uint32_t>(target_mesh.Indices().size()); ++i)
            {
                unique_indices[i] = vertex_mapping[target_mesh.Index(i)];
            }

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
                for (uint32_t j = 0; (j < static_cast<uint32_t>(unique_indices.size())) && !found; j += 3)
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
                                // pos_uv_mesh and target_mesh have the same vertex order
                                const auto& ai_mesh_vertex = pos_uv_mesh.Vertex(pos_uv_mesh.Index(j + (k + l) % 3));
                                auto& vertex = texture_transfer_vertices[i + l];
                                vertex.pos = ai_mesh_vertex.pos;
                                vertex.ai_tc = ai_mesh_vertex.texcoord;
                                vertex.photo_tc = textured_mesh.Vertex(textured_mesh.Index(i + l)).texcoord;
                            }
                            found = true;
                            break;
                        }
                    }
                }

                assert(found);
            }
        }

        void TransferTexture(GpuCommandList& cmd_list, GpuBuffer& vb, const GpuTexture2D& photo_tex, uint32_t texture_size,
            GpuTexture2D& color_gpu_tex, GpuTexture2D& pos_gpu_tex)
        {
            auto* d3d12_cmd_list = cmd_list.NativeCommandList<ID3D12GraphicsCommandList>();

            color_gpu_tex = GpuTexture2D(gpu_system_, texture_size, texture_size, 1, ColorFmt, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COMMON, L"color_gpu_tex");
            GpuRenderTargetView color_rtv(gpu_system_, color_gpu_tex);

            pos_gpu_tex = GpuTexture2D(gpu_system_, texture_size, texture_size, 1, PositionFmt, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COMMON, L"pos_gpu_tex");
            GpuRenderTargetView pos_rtv(gpu_system_, pos_gpu_tex);

            color_gpu_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_RENDER_TARGET);
            pos_gpu_tex.Transition(cmd_list, D3D12_RESOURCE_STATE_RENDER_TARGET);

            const float clear_clr[] = {0, 0, 0, 0};
            d3d12_cmd_list->ClearRenderTargetView(color_rtv.CpuHandle(), clear_clr, 0, nullptr);
            d3d12_cmd_list->ClearRenderTargetView(pos_rtv.CpuHandle(), clear_clr, 0, nullptr);

            const GpuCommandList::VertexBufferBinding vb_bindings[] = {{&vb, 0, sizeof(TextureTransferVertexFormat)}};
            const uint32_t num_verts = static_cast<uint32_t>(vb.Size() / sizeof(TextureTransferVertexFormat));

            GpuShaderResourceView photo_srv(gpu_system_, photo_tex);

            const GpuShaderResourceView* srvs[] = {&photo_srv};
            const GpuCommandList::ShaderBinding shader_bindings[] = {
                {{}, {}, {}},
                {{}, srvs, {}},
            };

            const GpuRenderTargetView* rtvs[] = {&color_rtv, &pos_rtv};

            const D3D12_VIEWPORT viewports[] = {{0, 0, static_cast<float>(texture_size), static_cast<float>(texture_size), 0, 1}};
            const D3D12_RECT scissor_rcs[] = {{0, 0, static_cast<LONG>(texture_size), static_cast<LONG>(texture_size)}};

            cmd_list.Render(
                transfer_texture_pipeline_, vb_bindings, nullptr, num_verts, shader_bindings, rtvs, nullptr, viewports, scissor_rcs);
        }

        GpuTexture2D* DilateTexture(GpuCommandList& cmd_list, GpuTexture2D& tex, GpuTexture2D& tmp_tex)
        {
            constexpr uint32_t BlockDim = 16;

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

                const GpuShaderResourceView* srvs[] = {tex_srvs[src]};
                GpuUnorderedAccessView* uavs[] = {tex_uavs[dst]};
                const GpuCommandList::ShaderBinding shader_binding = {{}, srvs, uavs};
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
        PyObjectPtr mesh_generator_gen_pos_mesh_method_;
        PyObjectPtr mesh_generator_query_colors_method_;

        PyObjectPtr pil_module_;
        PyObjectPtr image_class_;
        PyObjectPtr image_frombuffer_method_;

        GpuRenderPipeline transfer_texture_pipeline_;
        GpuComputePipeline get_pos_list_pipeline_;
        GpuComputePipeline dilate_pipeline_;

        static constexpr DXGI_FORMAT ColorFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
        static constexpr DXGI_FORMAT PositionFmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
        static constexpr uint32_t DilateTimes = 4;
    };

    MeshGenerator::MeshGenerator(const std::filesystem::path& exe_dir, GpuSystem& gpu_system, PythonSystem& python_system)
        : impl_(std::make_unique<Impl>(exe_dir, gpu_system, python_system))
    {
    }

    MeshGenerator::~MeshGenerator() noexcept = default;

    MeshGenerator::MeshGenerator(MeshGenerator&& other) noexcept = default;
    MeshGenerator& MeshGenerator::operator=(MeshGenerator&& other) noexcept = default;

    Mesh MeshGenerator::Generate(std::span<const Texture> input_images, uint32_t texture_size,
        const MeshReconstruction::Result& recon_input, const std::filesystem::path& tmp_dir)
    {
        return impl_->Generate(std::move(input_images), texture_size, recon_input, tmp_dir);
    }
} // namespace AIHoloImager
