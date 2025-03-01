// Copyright (c) 2024-2025 Minmin Gong
//

#include "MeshGenerator.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <numbers>
#include <set>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuTexture.hpp"
#include "MarchingCubes.hpp"
#include "MeshSimp/MeshSimplification.hpp"
#include "TextureRecon/TextureReconstruction.hpp"
#include "Util/BoundingBox.hpp"

#include "CompiledShader/MeshGen/DilateCs.h"
#include "CompiledShader/MeshGen/GatherVolumeCs.h"
#include "CompiledShader/MeshGen/MergeTextureCs.h"
#include "CompiledShader/MeshGen/ScatterIndexCs.h"

namespace AIHoloImager
{
    class MeshGenerator::Impl
    {
    public:
        Impl(const std::filesystem::path& exe_dir, GpuSystem& gpu_system, PythonSystem& python_system)
            : exe_dir_(exe_dir), gpu_system_(gpu_system), python_system_(python_system), marching_cubes_(gpu_system_),
              texture_recon_(exe_dir_, gpu_system_)
        {
            mesh_generator_module_ = python_system_.Import("MeshGenerator");
            mesh_generator_class_ = python_system_.GetAttr(*mesh_generator_module_, "MeshGenerator");
            mesh_generator_ = python_system_.CallObject(*mesh_generator_class_);
            mesh_generator_gen_volume_method_ = python_system_.GetAttr(*mesh_generator_, "GenVolume");
            mesh_generator_resolution_method_ = python_system_.GetAttr(*mesh_generator_, "Resolution");
            mesh_generator_coords_method_ = python_system_.GetAttr(*mesh_generator_, "Coords");
            mesh_generator_density_features_method_ = python_system_.GetAttr(*mesh_generator_, "DensityFeatures");
            mesh_generator_deformation_features_method_ = python_system_.GetAttr(*mesh_generator_, "DeformationFeatures");
            mesh_generator_color_features_method_ = python_system_.GetAttr(*mesh_generator_, "ColorFeatures");

            {
                const ShaderInfo shader = {ScatterIndexCs_shader, 1, 1, 1};
                scatter_index_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const ShaderInfo shader = {GatherVolumeCs_shader, 1, 4, 2};
                gather_volume_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                const GpuStaticSampler trilinear_sampler(
                    {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear}, GpuStaticSampler::AddressMode::Clamp);

                merge_texture_cb_ = ConstantBuffer<MergeTextureConstantBuffer>(gpu_system_, 1, L"merge_texture_cb_");
                merge_texture_cb_->inv_scale = 1 / GridScale;

                const ShaderInfo shader = {MergeTextureCs_shader, 1, 2, 1};
                merge_texture_pipeline_ = GpuComputePipeline(gpu_system_, shader, std::span{&trilinear_sampler, 1});
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
            assert(input_images.size() == NumMvImages);
            assert(input_images[0].Width() == MvImageDim);
            assert(input_images[0].Height() == MvImageDim);
            assert(FormatChannels(input_images[0].Format()) == MvImageChannels);

#ifdef AIHI_KEEP_INTERMEDIATES
            const auto output_dir = tmp_dir / "Texture";
            std::filesystem::create_directories(output_dir);
#endif

            std::cout << "Generating mesh from images...\n";

            GpuTexture3D color_vol_tex;
            Mesh mesh = this->GenMesh(input_images, color_vol_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshPosOnly.glb");
#endif

            std::cout << "Simplifying mesh...\n";

            MeshSimplification mesh_simp;
            mesh = mesh_simp.Process(mesh, 0.5f);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshSimplified.glb");
#endif

            Obb world_obb;
            const glm::mat4x4 model_mtx = this->CalcModelMatrix(mesh, recon_input, world_obb);

            mesh.ComputeNormals();

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshPosNormal.glb");
#endif

            std::cout << "Unwrapping UV...\n";

            mesh = mesh.UnwrapUv(texture_size, 2);

            std::cout << "Generating texture...\n";

            auto texture_result = texture_recon_.Process(mesh, model_mtx, world_obb, sfm_input, texture_size, tmp_dir);

            Texture merged_tex(texture_size, texture_size, ElementFormat::RGBA8_UNorm);
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                this->MergeTexture(cmd_list, color_vol_tex, texture_result.pos_tex, texture_result.inv_model, texture_result.color_tex);

                GpuTexture2D dilated_tmp_gpu_tex(gpu_system_, texture_result.color_tex.Width(0), texture_result.color_tex.Height(0), 1,
                    texture_result.color_tex.Format(), GpuResourceFlag::UnorderedAccess, L"dilated_tmp_tex");

                GpuTexture2D* dilated_gpu_tex = this->DilateTexture(cmd_list, texture_result.color_tex, dilated_tmp_gpu_tex);

                dilated_gpu_tex->Readback(gpu_system_, cmd_list, 0, merged_tex.Data());
                gpu_system_.Execute(std::move(cmd_list));
            }

            mesh.AlbedoTexture() = std::move(merged_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshTextured.glb");
#endif

            {
                const glm::mat4x4 adjust_mtx =
                    RegularizeTransform(-recon_input.obb.center, glm::inverse(recon_input.obb.orientation), glm::vec3(1)) * model_mtx;
                const glm::mat4x4 adjust_it_mtx = glm::transpose(glm::inverse(adjust_mtx));

                const uint32_t pos_attrib_index = mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Position, 0);
                const uint32_t normal_attrib_index = mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Normal, 0);
                for (uint32_t i = 0; i < mesh.NumVertices(); ++i)
                {
                    auto& pos = mesh.VertexData<glm::vec3>(i, pos_attrib_index);
                    const glm::vec4 p = adjust_mtx * glm::vec4(pos, 1);
                    pos = glm::vec3(p) / p.w;

                    auto& normal = mesh.VertexData<glm::vec3>(i, normal_attrib_index);
                    const glm::vec4 n = adjust_it_mtx * glm::vec4(normal, 0);
                    normal = glm::vec3(n);
                }
            }

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMesh.glb");
#endif

            return mesh;
        }

    private:
        Mesh GenMesh(std::span<const Texture> input_images, GpuTexture3D& color_tex)
        {
            auto args = python_system_.MakeTuple(1);
            {
                const uint32_t num_images = static_cast<uint32_t>(input_images.size());
                auto imgs_args = python_system_.MakeTuple(num_images);
                for (uint32_t i = 0; i < num_images; ++i)
                {
                    const auto& input_image = input_images[i];

                    auto py_image = python_system_.MakeTuple(4);

                    auto image_data = python_system_.MakeObject(
                        std::span<const std::byte>(reinterpret_cast<const std::byte*>(input_image.Data()), input_image.DataSize()));
                    python_system_.SetTupleItem(*py_image, 0, std::move(image_data));
                    python_system_.SetTupleItem(*py_image, 1, python_system_.MakeObject(input_image.Width()));
                    python_system_.SetTupleItem(*py_image, 2, python_system_.MakeObject(input_image.Height()));
                    python_system_.SetTupleItem(*py_image, 3, python_system_.MakeObject(FormatChannels(input_image.Format())));

                    python_system_.SetTupleItem(*imgs_args, i, std::move(py_image));
                }
                python_system_.SetTupleItem(*args, 0, std::move(imgs_args));
            }

            python_system_.CallObject(*mesh_generator_gen_volume_method_, *args);

            const auto py_grid_res = python_system_.CallObject(*mesh_generator_resolution_method_);
            const uint32_t grid_res = python_system_.Cast<uint32_t>(*py_grid_res);

            const auto py_coords = python_system_.CallObject(*mesh_generator_coords_method_);
            const auto coords = python_system_.ToSpan<const glm::uvec3>(*py_coords);

            const uint32_t size = grid_res + 1;

            auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            GpuBuffer coords_buff(gpu_system_, static_cast<uint32_t>(coords.size() * sizeof(glm::uvec3)), GpuHeap::Default,
                GpuResourceFlag::None, L"coords_buff");
            {
                GpuUploadBuffer coords_upload_buff(gpu_system_, coords_buff.Size(), L"coords_upload_buff");
                std::memcpy(coords_upload_buff.MappedData<glm::uvec3>(), coords.data(), coords_buff.Size());
                cmd_list.Copy(coords_buff, coords_upload_buff);
            }
            GpuShaderResourceView coords_srv(gpu_system_, coords_buff, GpuFormat::RGB32_Uint);

            GpuTexture3D index_vol_tex(
                gpu_system_, grid_res, grid_res, grid_res, 1, GpuFormat::R32_Uint, GpuResourceFlag::UnorderedAccess, L"index_vol_tex");
            {
                ConstantBuffer<ScatterIndexConstantBuffer> scatter_index_cb(gpu_system_, 1, L"scatter_index_cb");
                scatter_index_cb->num_features = static_cast<uint32_t>(coords.size());
                scatter_index_cb.UploadToGpu();

                GpuUnorderedAccessView index_vol_uav(gpu_system_, index_vol_tex);
                const uint32_t zeros[] = {0, 0, 0, 0};
                cmd_list.Clear(index_vol_uav, zeros);

                constexpr uint32_t BlockDim = 256;

                const GeneralConstantBuffer* cbs[] = {&scatter_index_cb};
                const GpuShaderResourceView* srvs[] = {&coords_srv};
                GpuUnorderedAccessView* uavs[] = {&index_vol_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};

                cmd_list.Compute(scatter_index_pipeline_, DivUp(static_cast<uint32_t>(coords.size()), BlockDim), 1, 1, shader_binding);
            }

            const auto py_density_features = python_system_.CallObject(*mesh_generator_density_features_method_);
            const auto density_features = python_system_.ToSpan<const float>(*py_density_features);

            const auto py_deformation_features = python_system_.CallObject(*mesh_generator_deformation_features_method_);
            const auto deformation_features = python_system_.ToSpan<const glm::vec3>(*py_deformation_features);

            const auto py_color_features = python_system_.CallObject(*mesh_generator_color_features_method_);
            const auto color_features = python_system_.ToSpan<const glm::vec3>(*py_color_features);

            GpuBuffer density_features_buff(gpu_system_, static_cast<uint32_t>(density_features.size() * sizeof(float)), GpuHeap::Default,
                GpuResourceFlag::None, L"density_features_buff");
            {
                GpuUploadBuffer density_features_upload_buff(gpu_system_, density_features_buff.Size(), L"density_features_upload_buff");
                std::memcpy(density_features_upload_buff.MappedData<float>(), density_features.data(), density_features_buff.Size());
                cmd_list.Copy(density_features_buff, density_features_upload_buff);
            }
            GpuShaderResourceView density_features_srv(gpu_system_, density_features_buff, GpuFormat::R32_Float);

            GpuBuffer deformation_features_buff(gpu_system_, static_cast<uint32_t>(deformation_features.size() * sizeof(glm::vec3)),
                GpuHeap::Default, GpuResourceFlag::None, L"deformation_features_buff");
            {
                GpuUploadBuffer deformation_features_upload_buff(
                    gpu_system_, deformation_features_buff.Size(), L"deformation_features_upload_buff");
                std::memcpy(deformation_features_upload_buff.MappedData<glm::vec3>(), deformation_features.data(),
                    deformation_features_buff.Size());
                cmd_list.Copy(deformation_features_buff, deformation_features_upload_buff);
            }
            GpuShaderResourceView deformation_features_srv(gpu_system_, deformation_features_buff, GpuFormat::RGB32_Float);

            GpuBuffer color_features_buff(gpu_system_, static_cast<uint32_t>(color_features.size() * sizeof(glm::vec3)), GpuHeap::Default,
                GpuResourceFlag::None, L"color_features_buff");
            {
                GpuUploadBuffer color_features_upload_buff(gpu_system_, color_features_buff.Size(), L"color_features_upload_buff");
                std::memcpy(color_features_upload_buff.MappedData<glm::vec3>(), color_features.data(), color_features_buff.Size());
                cmd_list.Copy(color_features_buff, color_features_upload_buff);
            }
            GpuShaderResourceView color_features_srv(gpu_system_, color_features_buff, GpuFormat::RGB32_Float);

            GpuTexture3D density_deformation_tex(
                gpu_system_, size, size, size, 1, GpuFormat::RGBA16_Float, GpuResourceFlag::UnorderedAccess, L"density_deformation_tex");
            color_tex =
                GpuTexture3D(gpu_system_, size, size, size, 1, GpuFormat::RGBA8_UNorm, GpuResourceFlag::UnorderedAccess, L"color_tex");

            {
                ConstantBuffer<GatherVolumeConstantBuffer> gather_volume_cb(gpu_system_, 1, L"gather_volume_cb");
                gather_volume_cb->grid_res = grid_res;
                gather_volume_cb->size = size;
                gather_volume_cb.UploadToGpu();

                GpuShaderResourceView index_vol_srv(gpu_system_, index_vol_tex);
                GpuUnorderedAccessView density_deformation_uav(gpu_system_, density_deformation_tex);
                GpuUnorderedAccessView color_uav(gpu_system_, color_tex);

                constexpr uint32_t BlockDim = 16;

                const GeneralConstantBuffer* cbs[] = {&gather_volume_cb};
                const GpuShaderResourceView* srvs[] = {
                    &index_vol_srv, &density_features_srv, &deformation_features_srv, &color_features_srv};
                GpuUnorderedAccessView* uavs[] = {&density_deformation_uav, &color_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};

                cmd_list.Compute(gather_volume_pipeline_, DivUp(size, BlockDim), DivUp(size, BlockDim), size, shader_binding);
            }

            gpu_system_.Execute(std::move(cmd_list));

            const Mesh pos_only_mesh = marching_cubes_.Generate(density_deformation_tex, 0, GridScale);
            return this->CleanMesh(pos_only_mesh);
        }

        Mesh CleanMesh(const Mesh& input_mesh)
        {
            constexpr float Scale = 1e5f;

            const auto& vertex_desc = input_mesh.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);
            std::set<std::array<int32_t, 3>> unique_int_pos;
            for (uint32_t i = 0; i < input_mesh.NumVertices(); ++i)
            {
                const glm::vec3& pos = input_mesh.VertexData<glm::vec3>(i, pos_attrib_index);
                std::array<int32_t, 3> int_pos = {static_cast<int32_t>(pos.x * Scale + 0.5f), static_cast<int32_t>(pos.y * Scale + 0.5f),
                    static_cast<int32_t>(pos.z * Scale + 0.5f)};
                unique_int_pos.emplace(std::move(int_pos));
            }

            Mesh ret_mesh(
                vertex_desc, static_cast<uint32_t>(unique_int_pos.size()), static_cast<uint32_t>(input_mesh.IndexBuffer().size()));

            std::vector<std::array<int32_t, 3>> unique_int_pos_vec(unique_int_pos.begin(), unique_int_pos.end());
            std::vector<uint32_t> vertex_mapping(input_mesh.NumVertices(), ~0U);
            const uint32_t vertex_size = vertex_desc.Stride();
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < input_mesh.NumVertices(); ++i)
            {
                const glm::vec3& pos = input_mesh.VertexData<glm::vec3>(i, 0);
                const std::array<int32_t, 3> int_pos = {static_cast<int32_t>(pos.x * Scale + 0.5f),
                    static_cast<int32_t>(pos.y * Scale + 0.5f), static_cast<int32_t>(pos.z * Scale + 0.5f)};

                const auto iter = std::lower_bound(unique_int_pos_vec.begin(), unique_int_pos_vec.end(), int_pos);
                assert(*iter == int_pos);

                if (vertex_mapping[i] == ~0U)
                {
                    vertex_mapping[i] = static_cast<uint32_t>(iter - unique_int_pos_vec.begin());
                    std::memcpy(ret_mesh.VertexDataPtr(vertex_mapping[i], 0), input_mesh.VertexDataPtr(i, 0), vertex_size);
                }
            }

            uint32_t num_faces = 0;
            for (uint32_t i = 0; i < static_cast<uint32_t>(input_mesh.IndexBuffer().size()); i += 3)
            {
                uint32_t face[3];
                for (uint32_t j = 0; j < 3; ++j)
                {
                    face[j] = vertex_mapping[input_mesh.Index(i + j)];
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
            std::vector<uint32_t> neighboring_face_indices(mesh.IndexBuffer().size());
            const auto mesh_indices = mesh.IndexBuffer();
            for (uint32_t i = 0; i < static_cast<uint32_t>(mesh.IndexBuffer().size() / 3); ++i)
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
            for (uint32_t i = 0; i < static_cast<uint32_t>(mesh.IndexBuffer().size() / 3); ++i)
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

                glm::vec3 bb_min(std::numeric_limits<float>::max());
                glm::vec3 bb_max(std::numeric_limits<float>::lowest());
                for (const uint32_t vi : new_component_vertices)
                {
                    const auto& pos = mesh.VertexData<glm::vec3>(vi, pos_attrib_index);
                    bb_min = glm::min(bb_min, pos);
                    bb_max = glm::max(bb_max, pos);
                }

                const glm::vec3 diagonal = bb_max - bb_min;
                const float bb_extent_sq = glm::dot(diagonal, diagonal);
                if (bb_extent_sq > largest_bb_extent_sq)
                {
                    largest_comp_face_indices.assign(new_component_faces.begin(), new_component_faces.end());
                    largest_bb_extent_sq = bb_extent_sq;
                }

                ++num_comps;
            }

            if (num_comps > 1)
            {
                const auto indices = mesh.IndexBuffer();
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

        void MergeTexture(GpuCommandList& cmd_list, const GpuTexture3D& color_vol_tex, const GpuTexture2D& pos_tex,
            const glm::mat4x4& inv_model, GpuTexture2D& color_tex)
        {
            GpuUnorderedAccessView merged_uav(gpu_system_, color_tex);

            const uint32_t texture_size = color_tex.Width(0);
            merge_texture_cb_->inv_model = glm::transpose(inv_model);
            merge_texture_cb_->texture_size = texture_size;
            merge_texture_cb_.UploadToGpu();

            GpuShaderResourceView pos_srv(gpu_system_, pos_tex);
            GpuShaderResourceView color_vol_srv(gpu_system_, color_vol_tex);

            constexpr uint32_t BlockDim = 16;

            const GeneralConstantBuffer* cbs[] = {&merge_texture_cb_};
            const GpuShaderResourceView* srvs[] = {&color_vol_srv, &pos_srv};
            GpuUnorderedAccessView* uavs[] = {&merged_uav};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(merge_texture_pipeline_, DivUp(texture_size, BlockDim), DivUp(texture_size, BlockDim), 1, shader_binding);
        }

        glm::mat4x4 CalcModelMatrix(const Mesh& mesh, const MeshReconstruction::Result& recon_input, Obb& world_obb)
        {
            glm::mat4x4 model_mtx =
                recon_input.transform * glm::rotate(glm::identity<glm::mat4x4>(), -std::numbers::pi_v<float> / 2, glm::vec3(1, 0, 0));

            const uint32_t pos_attrib_index = mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Position, 0);

            const Obb ai_obb =
                Obb::FromPoints(&mesh.VertexData<glm::vec3>(0, pos_attrib_index), mesh.MeshVertexDesc().Stride(), mesh.NumVertices());

            const Obb transformed_ai_obb = Obb::Transform(ai_obb, model_mtx);

            const float scale_x = transformed_ai_obb.extents.x / recon_input.obb.extents.x;
            const float scale_y = transformed_ai_obb.extents.y / recon_input.obb.extents.y;
            const float scale_z = transformed_ai_obb.extents.z / recon_input.obb.extents.z;
            const float scale = 1 / std::max({scale_x, scale_y, scale_z});

            model_mtx = glm::scale(model_mtx, glm::vec3(scale));
            world_obb = Obb::Transform(ai_obb, model_mtx);

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
        PyObjectPtr mesh_generator_gen_volume_method_;
        PyObjectPtr mesh_generator_resolution_method_;
        PyObjectPtr mesh_generator_coords_method_;
        PyObjectPtr mesh_generator_density_features_method_;
        PyObjectPtr mesh_generator_deformation_features_method_;
        PyObjectPtr mesh_generator_color_features_method_;

        MarchingCubes marching_cubes_;
        TextureReconstruction texture_recon_;

        struct ScatterIndexConstantBuffer
        {
            uint32_t num_features;
            uint32_t padding[3];
        };
        GpuComputePipeline scatter_index_pipeline_;

        struct GatherVolumeConstantBuffer
        {
            uint32_t grid_res;
            uint32_t size;
            uint32_t padding[2];
        };
        GpuComputePipeline gather_volume_pipeline_;

        struct MergeTextureConstantBuffer
        {
            glm::mat4x4 inv_model;

            uint32_t texture_size;
            float inv_scale;
            uint32_t padding[2];
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
        static constexpr uint32_t GridRes = 128;
        static constexpr float GridScale = 2.1f;
        static constexpr uint32_t NumMvImages = 6;
        static constexpr uint32_t MvImageDim = 320;
        static constexpr uint32_t MvImageChannels = 3;
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
