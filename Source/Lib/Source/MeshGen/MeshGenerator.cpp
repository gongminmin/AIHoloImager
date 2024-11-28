// Copyright (c) 2024 Minmin Gong
//

#include "MeshGenerator.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <numbers>
#include <set>

#include <directx/d3d12.h>
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

#include "CompiledShader/MeshGen/DeformationNnCs.h"
#include "CompiledShader/MeshGen/DensityNnCs.h"
#include "CompiledShader/MeshGen/DilateCs.h"
#include "CompiledShader/MeshGen/MergeTextureCs.h"

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
            mesh_generator_gen_nerf_method_ = python_system_.GetAttr(*mesh_generator_, "GenNeRF");

            const GpuStaticSampler bilinear_sampler(
                {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear}, GpuStaticSampler::AddressMode::Border);

            auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
            {
                PyObjectPtr num_density_nn_layers_method = python_system_.GetAttr(*mesh_generator_, "NumDensityNnLayers");
                const auto py_num_layers = python_system_.CallObject(*num_density_nn_layers_method);
                const uint32_t num_layers = python_system_.Cast<uint32_t>(*py_num_layers);

                density_nn_.resize(num_layers);

                PyObjectPtr density_nn_size_method = python_system_.GetAttr(*mesh_generator_, "DensityNnSize");
                PyObjectPtr density_nn_weight_method = python_system_.GetAttr(*mesh_generator_, "DensityNnWeight");
                PyObjectPtr density_nn_bias_method = python_system_.GetAttr(*mesh_generator_, "DensityNnBias");
                for (uint32_t i = 0; i < num_layers; ++i)
                {
                    auto layer_args = python_system_.MakeTuple(1);
                    python_system_.SetTupleItem(*layer_args, 0, python_system_.MakeObject(i));

                    const auto py_size = python_system_.CallObject(*density_nn_size_method, *layer_args);

                    NnLayer& layer = density_nn_[i];
                    layer.input_features = python_system_.Cast<uint32_t>(*python_system_.GetTupleItem(*py_size, 1));
                    layer.output_features = python_system_.Cast<uint32_t>(*python_system_.GetTupleItem(*py_size, 0));

                    layer.weight_buff = GpuBuffer(gpu_system_, layer.input_features * layer.output_features * sizeof(float),
                        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, std::format(L"layer {} weight_buff", i));
                    {
                        const auto py_weight = python_system_.CallObject(*density_nn_weight_method, *layer_args);

                        GpuUploadBuffer weight_upload_buff(gpu_system_, layer.weight_buff.Size(), L"weight_upload_buff");
                        std::memcpy(weight_upload_buff.MappedData<float>(), python_system_.ToSpan<const float>(*py_weight).data(),
                            layer.weight_buff.Size());
                        cmd_list.Copy(layer.weight_buff, weight_upload_buff);
                    }
                    layer.weight_srv = GpuShaderResourceView(gpu_system_, layer.weight_buff, DXGI_FORMAT_R32_FLOAT);

                    layer.bias_buff = GpuBuffer(gpu_system_, layer.output_features * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_FLAG_NONE, std::format(L"layer {} bias_buff", i));
                    {
                        const auto py_bias = python_system_.CallObject(*density_nn_bias_method, *layer_args);

                        GpuUploadBuffer bias_upload_buff(gpu_system_, layer.bias_buff.Size(), L"bias_upload_buff");
                        std::memcpy(bias_upload_buff.MappedData<float>(), python_system_.ToSpan<const float>(*py_bias).data(),
                            layer.bias_buff.Size());
                        cmd_list.Copy(layer.bias_buff, bias_upload_buff);
                    }
                    layer.bias_srv = GpuShaderResourceView(gpu_system_, layer.bias_buff, DXGI_FORMAT_R32_FLOAT);
                }
                {
                    density_nn_cb_ = ConstantBuffer<DensityNnConstantBuffer>(gpu_system_, 1, L"density_nn_cb_");

                    const ShaderInfo shader = {DensityNnCs_shader, 1, 9, 1};
                    density_nn_pipeline_ = GpuComputePipeline(gpu_system_, shader, std::span{&bilinear_sampler, 1});

                    density_nn_cb_->num_features = density_nn_[0].input_features;
                    density_nn_cb_->layer_1_nodes = density_nn_[0].output_features;
                    density_nn_cb_->layer_2_nodes = density_nn_[1].output_features;
                    density_nn_cb_->layer_3_nodes = density_nn_[2].output_features;
                    density_nn_cb_->layer_4_nodes = density_nn_[3].output_features;
                    density_nn_cb_->grid_res = GridRes;
                    density_nn_cb_->size = GridRes + 1;
                    density_nn_cb_->grid_scale = GridScale;
                }
            }
            {
                PyObjectPtr num_deformation_nn_layers_method = python_system_.GetAttr(*mesh_generator_, "NumDeformationNnLayers");
                const auto py_num_layers = python_system_.CallObject(*num_deformation_nn_layers_method);
                const uint32_t num_layers = python_system_.Cast<uint32_t>(*py_num_layers);

                deformation_nn_.resize(num_layers);

                PyObjectPtr deformation_nn_size_method = python_system_.GetAttr(*mesh_generator_, "DeformationNnSize");
                PyObjectPtr deformation_nn_weight_method = python_system_.GetAttr(*mesh_generator_, "DeformationNnWeight");
                PyObjectPtr deformation_nn_bias_method = python_system_.GetAttr(*mesh_generator_, "DeformationNnBias");
                for (uint32_t i = 0; i < num_layers; ++i)
                {
                    auto layer_args = python_system_.MakeTuple(1);
                    python_system_.SetTupleItem(*layer_args, 0, python_system_.MakeObject(i));

                    const auto py_size = python_system_.CallObject(*deformation_nn_size_method, *layer_args);

                    NnLayer& layer = deformation_nn_[i];
                    layer.input_features = python_system_.Cast<uint32_t>(*python_system_.GetTupleItem(*py_size, 1));
                    layer.output_features = python_system_.Cast<uint32_t>(*python_system_.GetTupleItem(*py_size, 0));

                    layer.weight_buff = GpuBuffer(gpu_system_, layer.input_features * layer.output_features * sizeof(float),
                        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, std::format(L"layer {} weight_buff", i));
                    {
                        const auto py_weight = python_system_.CallObject(*deformation_nn_weight_method, *layer_args);

                        GpuUploadBuffer weight_upload_buff(gpu_system_, layer.weight_buff.Size(), L"weight_upload_buff");
                        std::memcpy(weight_upload_buff.MappedData<float>(), python_system_.ToSpan<const float>(*py_weight).data(),
                            layer.weight_buff.Size());
                        cmd_list.Copy(layer.weight_buff, weight_upload_buff);
                    }
                    layer.weight_srv = GpuShaderResourceView(gpu_system_, layer.weight_buff, DXGI_FORMAT_R32_FLOAT);

                    layer.bias_buff = GpuBuffer(gpu_system_, layer.output_features * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_FLAG_NONE, std::format(L"layer {} bias_buff", i));
                    {
                        const auto py_bias = python_system_.CallObject(*deformation_nn_bias_method, *layer_args);

                        GpuUploadBuffer bias_upload_buff(gpu_system_, layer.bias_buff.Size(), L"bias_upload_buff");
                        std::memcpy(bias_upload_buff.MappedData<float>(), python_system_.ToSpan<const float>(*py_bias).data(),
                            layer.bias_buff.Size());
                        cmd_list.Copy(layer.bias_buff, bias_upload_buff);
                    }
                    layer.bias_srv = GpuShaderResourceView(gpu_system_, layer.bias_buff, DXGI_FORMAT_R32_FLOAT);
                }
                {
                    deformation_nn_cb_ = ConstantBuffer<DeformationNnConstantBuffer>(gpu_system_, 1, L"deformation_nn_cb_");

                    const ShaderInfo shader = {DeformationNnCs_shader, 1, 9, 1};
                    deformation_nn_pipeline_ = GpuComputePipeline(gpu_system_, shader, std::span{&bilinear_sampler, 1});

                    deformation_nn_cb_->num_features = deformation_nn_[0].input_features;
                    deformation_nn_cb_->layer_1_nodes = deformation_nn_[0].output_features;
                    deformation_nn_cb_->layer_2_nodes = deformation_nn_[1].output_features;
                    deformation_nn_cb_->layer_3_nodes = deformation_nn_[2].output_features;
                    deformation_nn_cb_->layer_4_nodes = deformation_nn_[3].output_features;
                    deformation_nn_cb_->grid_res = GridRes;
                    deformation_nn_cb_->size = GridRes + 1;
                    deformation_nn_cb_->grid_scale = GridScale;
                }
            }
            {
                PyObjectPtr num_color_nn_layers_method = python_system_.GetAttr(*mesh_generator_, "NumColorNnLayers");
                const auto py_num_layers = python_system_.CallObject(*num_color_nn_layers_method);
                const uint32_t num_layers = python_system_.Cast<uint32_t>(*py_num_layers);

                color_nn_.resize(num_layers);

                PyObjectPtr color_nn_size_method = python_system_.GetAttr(*mesh_generator_, "ColorNnSize");
                PyObjectPtr color_nn_weight_method = python_system_.GetAttr(*mesh_generator_, "ColorNnWeight");
                PyObjectPtr color_nn_bias_method = python_system_.GetAttr(*mesh_generator_, "ColorNnBias");
                for (uint32_t i = 0; i < num_layers; ++i)
                {
                    auto layer_args = python_system_.MakeTuple(1);
                    python_system_.SetTupleItem(*layer_args, 0, python_system_.MakeObject(i));

                    const auto py_size = python_system_.CallObject(*color_nn_size_method, *layer_args);

                    NnLayer& layer = color_nn_[i];
                    layer.input_features = python_system_.Cast<uint32_t>(*python_system_.GetTupleItem(*py_size, 1));
                    layer.output_features = python_system_.Cast<uint32_t>(*python_system_.GetTupleItem(*py_size, 0));

                    layer.weight_buff = GpuBuffer(gpu_system_, layer.input_features * layer.output_features * sizeof(float),
                        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, std::format(L"layer {} weight_buff", i));
                    {
                        const auto py_weight = python_system_.CallObject(*color_nn_weight_method, *layer_args);

                        GpuUploadBuffer weight_upload_buff(gpu_system_, layer.weight_buff.Size(), L"weight_upload_buff");
                        std::memcpy(weight_upload_buff.MappedData<float>(), python_system_.ToSpan<const float>(*py_weight).data(),
                            layer.weight_buff.Size());
                        cmd_list.Copy(layer.weight_buff, weight_upload_buff);
                    }
                    layer.weight_srv = GpuShaderResourceView(gpu_system_, layer.weight_buff, DXGI_FORMAT_R32_FLOAT);

                    layer.bias_buff = GpuBuffer(gpu_system_, layer.output_features * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_FLAG_NONE, std::format(L"layer {} bias_buff", i));
                    {
                        const auto py_bias = python_system_.CallObject(*color_nn_bias_method, *layer_args);

                        GpuUploadBuffer bias_upload_buff(gpu_system_, layer.bias_buff.Size(), L"bias_upload_buff");
                        std::memcpy(bias_upload_buff.MappedData<float>(), python_system_.ToSpan<const float>(*py_bias).data(),
                            layer.bias_buff.Size());
                        cmd_list.Copy(layer.bias_buff, bias_upload_buff);
                    }
                    layer.bias_srv = GpuShaderResourceView(gpu_system_, layer.bias_buff, DXGI_FORMAT_R32_FLOAT);
                }
                {
                    merge_texture_cb_ = ConstantBuffer<MergeTextureConstantBuffer>(gpu_system_, 1, L"merge_texture_cb_");

                    const ShaderInfo shader = {MergeTextureCs_shader, 1, 10, 1};
                    merge_texture_pipeline_ = GpuComputePipeline(gpu_system_, shader, std::span{&bilinear_sampler, 1});

                    merge_texture_cb_->num_features = color_nn_[0].input_features;
                    merge_texture_cb_->layer_1_nodes = color_nn_[0].output_features;
                    merge_texture_cb_->layer_2_nodes = color_nn_[1].output_features;
                    merge_texture_cb_->layer_3_nodes = color_nn_[2].output_features;
                    merge_texture_cb_->layer_4_nodes = color_nn_[3].output_features;
                }
            }
            gpu_system_.Execute(std::move(cmd_list));

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
            assert(input_images[0].NumChannels() == MvImageChannels);

#ifdef AIHI_KEEP_INTERMEDIATES
            const auto output_dir = tmp_dir / "Texture";
            std::filesystem::create_directories(output_dir);
#endif

            std::cout << "Generating NeRF from images...\n";

            this->GenNeRF(input_images);

            std::cout << "Generating mesh...\n";

            Mesh mesh = this->GenMesh();

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

            Texture merged_tex(texture_size, texture_size, 4);
            {
                auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

                this->MergeTexture(cmd_list, texture_result.pos_tex, texture_result.inv_model, texture_result.color_tex);

                GpuTexture2D dilated_tmp_gpu_tex(gpu_system_, texture_result.color_tex.Width(0), texture_result.color_tex.Height(0), 1,
                    texture_result.color_tex.Format(), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, L"dilated_tmp_tex");

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
        void GenNeRF(std::span<const Texture> input_images)
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

            const auto py_planes = python_system_.CallObject(*mesh_generator_gen_nerf_method_, *args);
            num_planes_ = python_system_.Cast<uint32_t>(*python_system_.GetTupleItem(*py_planes, 0));
            num_per_plane_features_ = python_system_.Cast<uint32_t>(*python_system_.GetTupleItem(*py_planes, 1));
            const uint32_t plane_width = python_system_.Cast<uint32_t>(*python_system_.GetTupleItem(*py_planes, 2));
            const uint32_t plane_height = python_system_.Cast<uint32_t>(*python_system_.GetTupleItem(*py_planes, 3));
            const auto planes = python_system_.ToSpan<const float>(*python_system_.GetTupleItem(*py_planes, 4));

            planes_tex_ = GpuTexture2DArray(gpu_system_, plane_width, plane_height, num_planes_ * num_per_plane_features_, 1,
                DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE, L"planes_tex_");
            planes_srv_ = GpuShaderResourceView(gpu_system_, planes_tex_);

            const float* data = planes.data();
            auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
            for (uint32_t i = 0; i < num_planes_ * num_per_plane_features_; ++i)
            {
                planes_tex_.Upload(gpu_system_, cmd_list, i, data);
                data += plane_width * plane_height;
            }
            gpu_system_.Execute(std::move(cmd_list));
        }

        Mesh GenMesh()
        {
            const uint32_t num_samples = (GridRes + 1) * (GridRes + 1) * (GridRes + 1);
            GpuTexture3D density_deformation_tex(gpu_system_, GridRes + 1, GridRes + 1, GridRes + 1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, L"density_deformation_tex");
            GpuUnorderedAccessView density_deformation_uav(gpu_system_, density_deformation_tex);

            auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
            {
                density_nn_cb_->num_samples = num_samples;
                assert(density_nn_cb_->num_features == num_planes_ * num_per_plane_features_);
                density_nn_cb_.UploadToGpu();

                constexpr uint32_t BlockDim = 256;

                const GeneralConstantBuffer* cbs[] = {&density_nn_cb_};
                const GpuShaderResourceView* srvs[] = {&planes_srv_, &density_nn_[0].weight_srv, &density_nn_[0].bias_srv,
                    &density_nn_[1].weight_srv, &density_nn_[1].bias_srv, &density_nn_[2].weight_srv, &density_nn_[2].bias_srv,
                    &density_nn_[3].weight_srv, &density_nn_[3].bias_srv};
                GpuUnorderedAccessView* uavs[] = {&density_deformation_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(density_nn_pipeline_, DivUp(num_samples, BlockDim), 1, 1, shader_binding);
            }
            {
                deformation_nn_cb_->num_samples = num_samples;
                assert(deformation_nn_cb_->num_features == num_planes_ * num_per_plane_features_);
                deformation_nn_cb_.UploadToGpu();

                constexpr uint32_t BlockDim = 256;

                const GeneralConstantBuffer* cbs[] = {&deformation_nn_cb_};
                const GpuShaderResourceView* srvs[] = {&planes_srv_, &deformation_nn_[0].weight_srv, &deformation_nn_[0].bias_srv,
                    &deformation_nn_[1].weight_srv, &deformation_nn_[1].bias_srv, &deformation_nn_[2].weight_srv,
                    &deformation_nn_[2].bias_srv, &deformation_nn_[3].weight_srv, &deformation_nn_[3].bias_srv};
                GpuUnorderedAccessView* uavs[] = {&density_deformation_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(deformation_nn_pipeline_, DivUp(num_samples, BlockDim), 1, 1, shader_binding);
            }
            gpu_system_.Execute(std::move(cmd_list));

            Mesh pos_only_mesh = marching_cubes_.Generate(density_deformation_tex, 0, GridScale);
            return this->CleanMesh(pos_only_mesh);
        }

        Mesh CleanMesh(const Mesh& input_mesh)
        {
            constexpr float Scale = 1e5f;

            uint32_t pos_attrib_index = input_mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Position, 0);
            std::set<std::array<int32_t, 3>> unique_int_pos;
            for (uint32_t i = 0; i < input_mesh.NumVertices(); ++i)
            {
                const glm::vec3& pos = input_mesh.VertexData<glm::vec3>(i, pos_attrib_index);
                std::array<int32_t, 3> int_pos = {static_cast<int32_t>(pos.x * Scale + 0.5f), static_cast<int32_t>(pos.y * Scale + 0.5f),
                    static_cast<int32_t>(pos.z * Scale + 0.5f)};
                unique_int_pos.emplace(std::move(int_pos));
            }

            const VertexAttrib pos_only_vertex_attribs[] = {
                {VertexAttrib::Semantic::Position, 0, 3},
            };
            Mesh ret_mesh(VertexDesc(pos_only_vertex_attribs), static_cast<uint32_t>(unique_int_pos.size()),
                static_cast<uint32_t>(input_mesh.IndexBuffer().size()));

            pos_attrib_index = ret_mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Position, 0);
            std::vector<std::array<int32_t, 3>> unique_int_pos_vec(unique_int_pos.begin(), unique_int_pos.end());
            std::vector<uint32_t> vertex_mapping(input_mesh.NumVertices(), ~0U);
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
                    ret_mesh.VertexData<glm::vec3>(vertex_mapping[i], pos_attrib_index) = pos;
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

        void MergeTexture(GpuCommandList& cmd_list, const GpuTexture2D& pos_tex, const glm::mat4x4& inv_model, GpuTexture2D& color_tex)
        {
            GpuUnorderedAccessView merged_uav(gpu_system_, color_tex);

            const uint32_t texture_size = color_tex.Width(0);
            merge_texture_cb_->inv_model = glm::transpose(inv_model);
            merge_texture_cb_->texture_size = texture_size;
            merge_texture_cb_.UploadToGpu();

            GpuShaderResourceView pos_srv(gpu_system_, pos_tex);

            constexpr uint32_t BlockDim = 16;

            const GeneralConstantBuffer* cbs[] = {&merge_texture_cb_};
            const GpuShaderResourceView* srvs[] = {&planes_srv_, &color_nn_[0].weight_srv, &color_nn_[0].bias_srv, &color_nn_[1].weight_srv,
                &color_nn_[1].bias_srv, &color_nn_[2].weight_srv, &color_nn_[2].bias_srv, &color_nn_[3].weight_srv, &color_nn_[3].bias_srv,
                &pos_srv};
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
        PyObjectPtr mesh_generator_gen_nerf_method_;

        MarchingCubes marching_cubes_;
        TextureReconstruction texture_recon_;

        struct NnLayer
        {
            uint32_t input_features;
            uint32_t output_features;
            GpuBuffer weight_buff;
            GpuShaderResourceView weight_srv;
            GpuBuffer bias_buff;
            GpuShaderResourceView bias_srv;
        };
        std::vector<NnLayer> density_nn_;
        std::vector<NnLayer> deformation_nn_;
        std::vector<NnLayer> color_nn_;

        GpuTexture2DArray planes_tex_;
        GpuShaderResourceView planes_srv_;
        uint32_t num_planes_;
        uint32_t num_per_plane_features_;

        struct DensityNnConstantBuffer
        {
            uint32_t num_samples;
            uint32_t num_features;

            uint32_t layer_1_nodes;
            uint32_t layer_2_nodes;
            uint32_t layer_3_nodes;
            uint32_t layer_4_nodes;

            uint32_t grid_res;
            uint32_t size;
            float grid_scale;
            uint32_t padding[3];
        };
        ConstantBuffer<DensityNnConstantBuffer> density_nn_cb_;
        GpuComputePipeline density_nn_pipeline_;

        struct DeformationNnConstantBuffer
        {
            uint32_t num_samples;
            uint32_t num_features;

            uint32_t layer_1_nodes;
            uint32_t layer_2_nodes;
            uint32_t layer_3_nodes;
            uint32_t layer_4_nodes;

            uint32_t grid_res;
            uint32_t size;
            float grid_scale;
            uint32_t padding[3];
        };
        ConstantBuffer<DeformationNnConstantBuffer> deformation_nn_cb_;
        GpuComputePipeline deformation_nn_pipeline_;

        struct MergeTextureConstantBuffer
        {
            uint32_t num_samples;
            uint32_t num_features;

            uint32_t layer_1_nodes;
            uint32_t layer_2_nodes;
            uint32_t layer_3_nodes;
            uint32_t layer_4_nodes;

            uint32_t texture_size;
            uint32_t padding;

            glm::mat4x4 inv_model;
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
