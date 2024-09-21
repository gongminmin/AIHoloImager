// Copyright (c) 2024 Minmin Gong
//

#include "MeshGenerator.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <set>

#include <directx/d3d12.h>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuTexture.hpp"
#include "MeshSimp/MeshSimplification.hpp"
#include "TextureRecon/TextureReconstruction.hpp"

#include "CompiledShader/DilateCs.h"
#include "CompiledShader/MergeTextureCs.h"

using namespace DirectX;

namespace AIHoloImager
{
    class MeshGenerator::Impl
    {
    public:
        Impl(const std::filesystem::path& exe_dir, GpuSystem& gpu_system, PythonSystem& python_system)
            : exe_dir_(exe_dir), gpu_system_(gpu_system), python_system_(python_system), texture_recon_(exe_dir_, gpu_system_)
        {
            mesh_generator_module_ = python_system_.Import("MeshGenerator");
            mesh_generator_class_ = python_system_.GetAttr(*mesh_generator_module_, "MeshGenerator");
            mesh_generator_ = python_system_.CallObject(*mesh_generator_class_);
            mesh_generator_gen_nerf_method_ = python_system_.GetAttr(*mesh_generator_, "GenNeRF");
            mesh_generator_gen_pos_mesh_method_ = python_system_.GetAttr(*mesh_generator_, "GenPosMesh");
            mesh_generator_query_colors_method_ = python_system_.GetAttr(*mesh_generator_, "QueryColors");

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

        ~Impl()
        {
            auto mesh_generator_destroy_method = python_system_.GetAttr(*mesh_generator_, "Destroy");
            python_system_.CallObject(*mesh_generator_destroy_method);
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

            Mesh mesh = this->GenMeshFromImages(input_images);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshPosOnly.glb");
#endif

            std::cout << "Simplifying mesh...\n";

            MeshSimplification mesh_simp;
            mesh = mesh_simp.Process(mesh, 0.5f);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshSimplified.glb");
#endif

            BoundingOrientedBox world_obb;
            const XMMATRIX model_mtx = this->CalcModelMatrix(mesh, recon_input, world_obb);

            mesh.ComputeNormals();

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshPosNormal.glb");
#endif

            std::cout << "Unwrapping UV...\n";

            mesh = mesh.UnwrapUv(texture_size, 2);

            std::cout << "Generating texture...\n";

            auto texture_result = texture_recon_.Process(mesh, model_mtx, world_obb, sfm_input, texture_size, true, tmp_dir);

            GpuReadbackBuffer counter_cpu_buff;
            GpuReadbackBuffer pos_cpu_buff;
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

#ifdef AIHI_KEEP_INTERMEDIATES
                {
                    SaveMesh(mesh, output_dir / "AiMeshTextured.glb");

                    Texture projective_tex(texture_size, texture_size, 4);
                    texture_result.color_tex.Readback(gpu_system_, cmd_list, 0, projective_tex.Data());
                    SaveTexture(projective_tex, output_dir / "Projective.png");
                }
#endif

                counter_cpu_buff = GpuReadbackBuffer(gpu_system_, texture_result.counter_buff.Size(), L"counter_cpu_buff");
                cmd_list.Copy(counter_cpu_buff, texture_result.counter_buff);
                texture_result.counter_buff = GpuBuffer();

                pos_cpu_buff = GpuReadbackBuffer(gpu_system_, texture_result.pos_buff.Size(), L"pos_cpu_buff");
                cmd_list.Copy(pos_cpu_buff, texture_result.pos_buff);
                texture_result.pos_buff = GpuBuffer();

                gpu_system_.Execute(std::move(cmd_list));
            }

            {
                const XMVECTOR center = XMLoadFloat3(&recon_input.obb.Center);
                const XMMATRIX pre_trans = XMMatrixTranslationFromVector(-center);
                const XMMATRIX pre_rotate = XMMatrixRotationQuaternion(XMQuaternionInverse(XMLoadFloat4(&recon_input.obb.Orientation))) *
                                            XMMatrixRotationZ(XM_PI / 2) * XMMatrixRotationX(XM_PI);
                const XMMATRIX handedness = XMMatrixScaling(1, 1, -1);

                const XMMATRIX adjust_mtx = model_mtx * handedness * pre_trans * pre_rotate * handedness;
                const XMMATRIX adjust_it_mtx = XMMatrixTranspose(XMMatrixInverse(nullptr, adjust_mtx));

                const uint32_t pos_attrib_index = mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Position, 0);
                const uint32_t normal_attrib_index = mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Normal, 0);
                for (uint32_t i = 0; i < mesh.NumVertices(); ++i)
                {
                    auto& pos = mesh.VertexData<XMFLOAT3>(i, pos_attrib_index);
                    XMStoreFloat3(&pos, XMVector3TransformCoord(XMLoadFloat3(&pos), adjust_mtx));

                    auto& normal = mesh.VertexData<XMFLOAT3>(i, normal_attrib_index);
                    XMStoreFloat3(&normal, XMVector3TransformNormal(XMLoadFloat3(&normal), adjust_it_mtx));
                }
            }
            gpu_system_.WaitForGpu();

            this->MergeTexture(counter_cpu_buff, texture_result.uv_buff, pos_cpu_buff, texture_result.color_tex);

            Texture merged_tex(texture_size, texture_size, 4);
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                GpuTexture2D dilated_tmp_gpu_tex(gpu_system_, texture_result.color_tex.Width(0), texture_result.color_tex.Height(0), 1,
                    texture_result.color_tex.Format(), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
                    L"dilated_tmp_tex");

                GpuTexture2D* dilated_gpu_tex = this->DilateTexture(cmd_list, texture_result.color_tex, dilated_tmp_gpu_tex);

                dilated_gpu_tex->Readback(gpu_system_, cmd_list, 0, merged_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));
            }

            mesh.AlbedoTexture() = std::move(merged_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMesh.glb");
#endif

            return mesh;
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

            const VertexAttrib pos_only_vertex_attribs[] = {
                {VertexAttrib::Semantic::Position, 0, 3},
            };
            Mesh ret_mesh(
                VertexDesc(pos_only_vertex_attribs), static_cast<uint32_t>(unique_int_pos.size()), static_cast<uint32_t>(indices.size()));

            const uint32_t pos_attrib_index = ret_mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Position, 0);
            std::vector<std::array<int32_t, 3>> unique_int_pos_vec(unique_int_pos.begin(), unique_int_pos.end());
            std::vector<uint32_t> vertex_mapping(positions.size(), ~0U);
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

                if (vertex_mapping[i] == ~0U)
                {
                    vertex_mapping[i] = static_cast<uint32_t>(iter - unique_int_pos_vec.begin());
                    ret_mesh.VertexData<XMFLOAT3>(vertex_mapping[i], pos_attrib_index) = pos;
                }
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
            const uint32_t pos_attrib_index = mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Position, 0);

            std::vector<uint32_t> num_neighboring_faces(mesh.NumVertices(), 0);
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

            std::vector<uint32_t> base_neighboring_faces(mesh.NumVertices() + 1);
            base_neighboring_faces[0] = 0;
            for (size_t i = 1; i < mesh.NumVertices() + 1; ++i)
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

            std::vector<bool> vertex_occupied(mesh.NumVertices(), false);
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
                    const auto& vert = mesh.VertexData<XMFLOAT3>(vi, pos_attrib_index);
                    const XMVECTOR pos = XMLoadFloat3(&vert);
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
                std::vector<uint32_t> extract_indices(largest_comp_face_indices.size() * 3);
                for (uint32_t i = 0; i < largest_comp_face_indices.size(); ++i)
                {
                    for (uint32_t j = 0; j < 3; ++j)
                    {
                        extract_indices[i * 3 + j] = indices[largest_comp_face_indices[i] * 3 + j];
                    }
                }

                mesh = mesh.ExtractMesh(mesh.MeshVertexDesc(), extract_indices);
            }
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

        XMMATRIX CalcModelMatrix(const Mesh& mesh, const MeshReconstruction::Result& recon_input, BoundingOrientedBox& world_obb)
        {
            XMMATRIX model_mtx = XMLoadFloat4x4(&recon_input.transform);
            model_mtx *= XMMatrixScaling(1, 1, -1);    // RH to LH
            std::swap(model_mtx.r[1], model_mtx.r[2]); // Swap Y and Z

            const uint32_t pos_attrib_index = mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Position, 0);

            BoundingOrientedBox ai_obb;
            BoundingOrientedBox::CreateFromPoints(
                ai_obb, mesh.NumVertices(), &mesh.VertexData<XMFLOAT3>(0, pos_attrib_index), mesh.MeshVertexDesc().Stride());

            BoundingOrientedBox transformed_ai_obb;
            ai_obb.Transform(transformed_ai_obb, model_mtx);

            const float scale_x = transformed_ai_obb.Extents.x / recon_input.obb.Extents.x;
            const float scale_y = transformed_ai_obb.Extents.y / recon_input.obb.Extents.y;
            const float scale_z = transformed_ai_obb.Extents.z / recon_input.obb.Extents.z;
            const float scale = 1 / std::max({scale_x, scale_y, scale_z});

            model_mtx = XMMatrixScaling(scale, scale, scale) * model_mtx;
            ai_obb.Transform(world_obb, model_mtx);

            return model_mtx;
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

        TextureReconstruction texture_recon_;

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
