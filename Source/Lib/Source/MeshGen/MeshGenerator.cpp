// Copyright (c) 2024-2025 Minmin Gong
//

#include "MeshGenerator.hpp"

#include <array>
#include <cassert>
#include <future>
#include <iostream>
#include <map>
#include <numbers>
#include <set>
#include <tuple>
#include <type_traits>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 5054) // Ignore operator between enums of different types
#endif
#include <Eigen/Core>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif
#include <Eigen/Eigenvalues>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#ifndef GLM_ENABLE_EXPERIMENTAL
    #define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/vec3.hpp>

#include "Base/Timer.hpp"
#include "DiffOptimizer.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuTexture.hpp"
#include "InvisibleFacesRemover.hpp"
#include "MarchingCubes.hpp"
#include "MeshSimp/MeshSimplification.hpp"
#include "TextureRecon/TextureReconstruction.hpp"
#include "Util/BoundingBox.hpp"
#include "Util/FormatConversion.hpp"

#include "CompiledShader/MeshGen/ApplyVertexColorCs.h"
#include "CompiledShader/MeshGen/Dilate3DCs.h"
#include "CompiledShader/MeshGen/DilateCs.h"
#include "CompiledShader/MeshGen/GatherVolumeCs.h"
#include "CompiledShader/MeshGen/MergeTextureCs.h"
#include "CompiledShader/MeshGen/ResizeCs.h"
#include "CompiledShader/MeshGen/RotatePs.h"
#include "CompiledShader/MeshGen/RotateVs.h"
#include "CompiledShader/MeshGen/ScatterIndexCs.h"

namespace AIHoloImager
{
    void TransformMesh(Mesh& mesh, const glm::mat4x4& mtx)
    {
        const glm::mat3x3 mtx_it = glm::transpose(glm::inverse(mtx));

        const auto& vertex_desc = mesh.MeshVertexDesc();
        const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);
        const uint32_t normal_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Normal, 0);
        for (uint32_t i = 0; i < mesh.NumVertices(); ++i)
        {
            if (pos_attrib_index != VertexDesc::InvalidIndex)
            {
                auto& pos = mesh.VertexData<glm::vec3>(i, pos_attrib_index);
                const glm::vec4 p = mtx * glm::vec4(pos, 1);
                pos = glm::vec3(p) / p.w;
            }
            if (normal_attrib_index != VertexDesc::InvalidIndex)
            {
                auto& normal = mesh.VertexData<glm::vec3>(i, normal_attrib_index);
                normal = mtx_it * normal;
            }
        }
    }

    class MeshGenerator::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi), invisible_faces_remover_(aihi), marching_cubes_(aihi), texture_recon_(aihi)
        {
            Timer timer;

            auto& gpu_system = aihi_.GpuSystemInstance();

            py_init_future_ = std::async(std::launch::async, [this] {
                Timer timer;

                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();

                mesh_generator_module_ = python_system.Import("MeshGenerator");
                mesh_generator_class_ = python_system.GetAttr(*mesh_generator_module_, "MeshGenerator");
                mesh_generator_ = python_system.CallObject(*mesh_generator_class_);
                mesh_generator_gen_features_method_ = python_system.GetAttr(*mesh_generator_, "GenFeatures");
                mesh_generator_resolution_method_ = python_system.GetAttr(*mesh_generator_, "Resolution");
                mesh_generator_coords_method_ = python_system.GetAttr(*mesh_generator_, "Coords");
                mesh_generator_density_features_method_ = python_system.GetAttr(*mesh_generator_, "DensityFeatures");
                mesh_generator_deformation_features_method_ = python_system.GetAttr(*mesh_generator_, "DeformationFeatures");
                mesh_generator_color_features_method_ = python_system.GetAttr(*mesh_generator_, "ColorFeatures");

                aihi_.AddTiming("Mesh generator init (async)", timer.Elapsed());
            });

            {
                const ShaderInfo shaders[] = {
                    {RotateVs_shader, 1, 0, 0},
                    {RotatePs_shader, 0, 1, 0},
                };

                const GpuFormat rtv_formats[] = {ColorFmt};

                GpuRenderPipeline::States states;
                states.cull_mode = GpuRenderPipeline::CullMode::CounterClockWise;
                states.rtv_formats = rtv_formats;

                const GpuStaticSampler bilinear_sampler(
                    {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear}, GpuStaticSampler::AddressMode::Clamp);

                rotate_pipeline_ = GpuRenderPipeline(gpu_system, GpuRenderPipeline::PrimitiveTopology::TriangleStrip, shaders,
                    GpuVertexAttribs({}), std::span(&bilinear_sampler, 1), states);
            }
            {
                const ShaderInfo shader = {ResizeCs_shader, 1, 1, 1};
                resize_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {ScatterIndexCs_shader, 1, 1, 1};
                scatter_index_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {GatherVolumeCs_shader, 1, 4, 2};
                gather_volume_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }

            const GpuStaticSampler trilinear_sampler(
                {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear}, GpuStaticSampler::AddressMode::Clamp);
            {
                merge_texture_cb_ = ConstantBuffer<MergeTextureConstantBuffer>(gpu_system, 1, L"merge_texture_cb_");
                merge_texture_cb_->inv_scale = 1 / GridScale;

                const ShaderInfo shader = {MergeTextureCs_shader, 1, 2, 1};
                merge_texture_pipeline_ = GpuComputePipeline(gpu_system, shader, std::span{&trilinear_sampler, 1});
            }
            {
                const ShaderInfo shader = {DilateCs_shader, 1, 1, 1};
                dilate_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {Dilate3DCs_shader, 1, 1, 1};
                dilate_3d_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                apply_vertex_color_cb_ = ConstantBuffer<ApplyVertexColorConstantBuffer>(gpu_system, 1, L"apply_vertex_color_cb_");
                apply_vertex_color_cb_->inv_scale = 1 / GridScale;

                const ShaderInfo shader = {ApplyVertexColorCs_shader, 1, 2, 1};
                apply_vertex_color_pipeline_ = GpuComputePipeline(gpu_system, shader, std::span{&trilinear_sampler, 1});
            }

            aihi_.AddTiming("Mesh generator init", timer.Elapsed());
        }

        ~Impl()
        {
            PythonSystem::GilGuard guard;

            auto& python_system = aihi_.PythonSystemInstance();
            auto mesh_generator_destroy_method = python_system.GetAttr(*mesh_generator_, "Destroy");
            python_system.CallObject(*mesh_generator_destroy_method);

            mesh_generator_destroy_method.reset();
            mesh_generator_color_features_method_.reset();
            mesh_generator_deformation_features_method_.reset();
            mesh_generator_density_features_method_.reset();
            mesh_generator_coords_method_.reset();
            mesh_generator_resolution_method_.reset();
            mesh_generator_gen_features_method_.reset();
            mesh_generator_.reset();
            mesh_generator_class_.reset();
            mesh_generator_module_.reset();
        }

        Mesh Generate(const StructureFromMotion::Result& sfm_input, uint32_t texture_size, const std::filesystem::path& tmp_dir)
        {
            Timer timer;

#ifdef AIHI_KEEP_INTERMEDIATES
            const auto output_dir = tmp_dir / "MeshGen";
            std::filesystem::create_directories(output_dir);
#endif

            std::cout << "Rotating images...\n";

            const glm::vec3 centroid = this->ForegroundObjectCentroid(sfm_input);
            const glm::vec3 up_vec = this->SceneUpVector(sfm_input, centroid);
            std::vector<Texture> rotated_images = this->RotateImages(sfm_input, up_vec);

#ifdef AIHI_KEEP_INTERMEDIATES
            for (size_t i = 0; i < rotated_images.size(); ++i)
            {
                SaveTexture(rotated_images[i], output_dir / std::format("Rotated_{}.png", i));
            }
#endif

            std::cout << "Generating mesh from images...\n";

            GpuTexture3D color_vol_tex;
            Mesh mesh = this->GenMesh(rotated_images, color_vol_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshPosOnly.glb");
#endif

            Mesh pos_color_mesh = this->ApplyVertexColor(mesh, color_vol_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(pos_color_mesh, output_dir / "AiMeshPosColor.glb");
#endif

            Obb obb;
            {
                const auto& vertex_desc = mesh.MeshVertexDesc();
                const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);
                obb = Obb::FromPoints(&mesh.VertexData<glm::vec3>(0, pos_attrib_index), vertex_desc.Stride(), mesh.NumVertices());
            }

            glm::vec3 local_up_vec;
            {
                const glm::vec3 local_y =
                    glm::rotate(glm::inverse(obb.orientation), glm::vec3(0, 0, 1)); // Y and Z are swapped in the output of TRELLIS
                const glm::vec3 abs_local_y = glm::abs(local_y);
                if (abs_local_y.x > abs_local_y.y)
                {
                    if (abs_local_y.x > abs_local_y.z)
                    {
                        local_up_vec = glm::vec3(local_y.x / abs_local_y.x, 0, 0);
                    }
                    else
                    {
                        local_up_vec = glm::vec3(0, 0, local_y.z / abs_local_y.z);
                    }
                }
                else
                {
                    if (local_y.y > abs_local_y.z)
                    {
                        local_up_vec = glm::vec3(0, local_y.y / abs_local_y.y, 0);
                    }
                    else
                    {
                        local_up_vec = glm::vec3(0, 0, local_y.z / abs_local_y.z);
                    }
                }

                local_up_vec = glm::rotate(obb.orientation, local_up_vec);
            }

            glm::mat4x4 model_mtx = this->GuessModelMatrix(sfm_input, obb, centroid, local_up_vec, up_vec);

            std::cout << "Optimizing transform...\n";

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                Mesh before_opt_mesh = mesh;
                TransformMesh(before_opt_mesh, model_mtx);
                SaveMesh(before_opt_mesh, output_dir / "BeforeOpt.glb");
            }
#endif

            {
                DiffOptimizer optimizer(aihi_);
                optimizer.Optimize(pos_color_mesh, model_mtx, sfm_input);
            }

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                Mesh after_opt_mesh = mesh;
                TransformMesh(after_opt_mesh, model_mtx);
                SaveMesh(after_opt_mesh, output_dir / "AfterOpt.glb");
            }
#endif

            std::cout << "Simplifying mesh...\n";

            MeshSimplification mesh_simp;
            mesh = mesh_simp.Process(mesh, 0.125f);
            this->FillHoles(mesh);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshSimplified.glb");
#endif

            mesh.ComputeNormals();

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshPosNormal.glb");
#endif

            std::cout << "Unwrapping UV...\n";

            mesh = mesh.UnwrapUv(texture_size, 2);

            std::cout << "Generating texture...\n";

            const Obb world_obb = Obb::Transform(obb, model_mtx);
            auto texture_result = texture_recon_.Process(mesh, model_mtx, world_obb, sfm_input, texture_size, tmp_dir);

            Texture merged_tex(texture_size, texture_size, ElementFormat::RGBA8_UNorm);
            {
                auto& gpu_system = aihi_.GpuSystemInstance();
                auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Render);

                this->MergeTexture(cmd_list, color_vol_tex, texture_result.pos_tex, model_mtx, texture_result.color_tex);

                GpuTexture2D dilated_tmp_gpu_tex(gpu_system, texture_result.color_tex.Width(0), texture_result.color_tex.Height(0), 1,
                    texture_result.color_tex.Format(), GpuResourceFlag::UnorderedAccess, L"dilated_tmp_tex");

                GpuTexture2D* dilated_gpu_tex = this->DilateTexture(cmd_list, texture_result.color_tex, dilated_tmp_gpu_tex);

                dilated_gpu_tex->ReadBack(gpu_system, cmd_list, 0, merged_tex.Data());
                gpu_system.Execute(std::move(cmd_list));
            }

            mesh.AlbedoTexture() = std::move(merged_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMeshTextured.glb");
#endif

            {
                glm::vec3 scale;
                glm::quat rotation;
                glm::vec3 translation;
                glm::vec3 skew;
                glm::vec4 perspective;
                glm::decompose(model_mtx, scale, rotation, translation, skew, perspective);

                const glm::mat4x4 adjust_mtx =
                    glm::recompose(scale, glm::rotation(local_up_vec, glm::vec3(0, 1, 0)), glm::zero<glm::vec3>(), skew, perspective);
                TransformMesh(mesh, adjust_mtx);
            }

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, output_dir / "AiMesh.glb");
#endif

            aihi_.AddTiming("Mesh generator process", timer.Elapsed());

            return mesh;
        }

    private:
        glm::vec4 FitPlane(const std::span<const glm::vec3> points)
        {
            // Step 1: Compute the centroid of the points
            glm::vec3 centroid(0.0f);
            for (const auto& point : points)
            {
                centroid += point;
            }
            centroid /= static_cast<float>(points.size());

            // Step 2: Compute the covariance matrix
            Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
            for (const auto& point : points)
            {
                const glm::vec3 centered = point - centroid;
                covariance(0, 0) += centered.x * centered.x;
                covariance(0, 1) += centered.x * centered.y;
                covariance(0, 2) += centered.x * centered.z;
                covariance(1, 0) += centered.y * centered.x;
                covariance(1, 1) += centered.y * centered.y;
                covariance(1, 2) += centered.y * centered.z;
                covariance(2, 0) += centered.z * centered.x;
                covariance(2, 1) += centered.z * centered.y;
                covariance(2, 2) += centered.z * centered.z;
            }
            covariance /= static_cast<float>(points.size());

            // Step 3: Perform eigenvalue decomposition
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
            const Eigen::Vector3f normal = solver.eigenvectors().col(0); // Smallest eigenvalue

            const glm::vec3 plane_normal = glm::normalize(glm::vec3(normal.x(), normal.y(), normal.z()));
            return glm::vec4(plane_normal, -glm::dot(centroid, plane_normal));
        }

        glm::vec3 ForegroundObjectCentroid(const StructureFromMotion::Result& sfm_input)
        {
            glm::vec3 centroid(0.0f);
            uint32_t num = 0;
            for (const auto& landmark : sfm_input.structure)
            {
                for (const auto& ob : landmark.obs)
                {
                    const uint32_t x = static_cast<uint32_t>(std::round(ob.point.x));
                    const uint32_t y = static_cast<uint32_t>(std::round(ob.point.y));

                    const auto& view = sfm_input.views[ob.view_id];
                    const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                    const std::byte* image_mask_data = view.image_mask.Data();
                    const uint32_t fmt_size = FormatSize(view.image_mask.Format());
                    if (image_mask_data[(y * intrinsic.width + x) * fmt_size + 3] > std::byte(0x7F))
                    {
                        centroid += landmark.point;
                        ++num;
                    }
                }
            }
            centroid /= static_cast<float>(num);

            return centroid;
        }

        glm::vec3 SceneUpVector(const StructureFromMotion::Result& sfm_input, const glm::vec3& centroid)
        {
            std::vector<bool> point_used(sfm_input.structure.size(), false);
            std::vector<glm::vec3> plane_points;
            for (uint32_t i = 0; i < sfm_input.views.size(); ++i)
            {
                const auto& view = sfm_input.views[i];
                const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                const std::byte* image_mask_data = view.image_mask.Data();
                const uint32_t fmt_size = FormatSize(view.image_mask.Format());

                constexpr uint32_t Gap = 32;
                const uint32_t beg_x = view.delighted_offset.x - Gap;
                const uint32_t beg_y = view.delighted_offset.y + view.delighted_image.Height() / 2;
                const uint32_t end_x = view.delighted_offset.x + view.delighted_image.Width() + Gap;
                const uint32_t end_y = view.delighted_offset.y + view.delighted_image.Height() + Gap;

                for (size_t j = 0; j < sfm_input.structure.size(); ++j)
                {
                    if (!point_used[j])
                    {
                        const auto& landmark = sfm_input.structure[j];
                        for (const auto& ob : landmark.obs)
                        {
                            if (ob.view_id == i)
                            {
                                const uint32_t x = static_cast<uint32_t>(std::round(ob.point.x));
                                const uint32_t y = static_cast<uint32_t>(std::round(ob.point.y));
                                if ((x >= beg_x) && (y >= beg_y) && (x < end_x) && (y < end_y))
                                {
                                    if (image_mask_data[(y * intrinsic.width + x) * fmt_size + 3] <= std::byte(0x7F))
                                    {
                                        plane_points.push_back(landmark.point);
                                        point_used[j] = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            glm::vec4 plane = this->FitPlane(plane_points);
            if (glm::dot(glm::vec3(plane), centroid) + plane.w < 0)
            {
                plane = -plane;
            }

            return glm::vec3(plane);
        }

        std::vector<Texture> RotateImages(const StructureFromMotion::Result& sfm_input, const glm::vec3& up_vec)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            const glm::vec4 target_up_vec_ws = glm::vec4(up_vec.x, up_vec.y, up_vec.z, 0);

            std::vector<Texture> rotated_images(sfm_input.views.size());
            for (uint32_t i = 0; i < sfm_input.views.size(); ++i)
            {
                const auto& view = sfm_input.views[i];

                auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Render);

                GpuTexture2D rotated_roi_tex;
                {
                    const uint32_t delighted_width = view.delighted_image.Width();
                    const uint32_t delighted_height = view.delighted_image.Height();
                    GpuTexture2D delighted_tex(gpu_system, delighted_width, delighted_height, 1, ToGpuFormat(view.delighted_image.Format()),
                        GpuResourceFlag::None, L"delighted_tex");
                    delighted_tex.Upload(gpu_system, cmd_list, 0, view.delighted_image.Data());
                    GpuShaderResourceView delighted_srv(gpu_system, delighted_tex);

                    ConstantBuffer<RotateConstantBuffer> rotation_cb(gpu_system, 1, L"rotation_cb");

                    const auto& view_mtx = CalcViewMatrix(view);
                    const glm::vec3 forward_vec(0, 0, 1);
                    const glm::vec3 target_up_vec = glm::normalize(glm::vec3(view_mtx * target_up_vec_ws));
                    const glm::vec3 old_right_vec(1, 0, 0);
                    const glm::vec3 new_right_vec = glm::cross(target_up_vec, forward_vec);

                    const glm::quat rotation = glm::normalize(glm::rotation(new_right_vec, old_right_vec));
                    rotation_cb->rotation_mtx = glm::transpose(glm::mat4_cast(rotation));

                    const uint32_t size = std::max(delighted_width, delighted_height);
                    const int32_t extent = size / 2;
                    const glm::ivec2 center = glm::ivec2(delighted_width, delighted_height) / 2;

                    const glm::vec2 top_left = center - extent;
                    const glm::vec2 bottom_right = center + extent;
                    const glm::vec2 wh(delighted_width, delighted_height);
                    rotation_cb->tc_bounding_box = glm::vec4(top_left / wh, bottom_right / wh);

                    rotation_cb.UploadToGpu();

                    const float cos_theta = glm::dot(new_right_vec, old_right_vec);
                    const float abs_sin_theta = std::sqrt(1 - cos_theta * cos_theta);
                    const uint32_t rotated_size = static_cast<uint32_t>(std::round(size * (std::abs(cos_theta) + abs_sin_theta)));
                    rotated_roi_tex = GpuTexture2D(
                        gpu_system, rotated_size, rotated_size, 1, ColorFmt, GpuResourceFlag::RenderTarget, L"rotated_roi_tex");
                    GpuRenderTargetView rotated_roi_rtv(gpu_system, rotated_roi_tex);

                    const float clear_clr[] = {0, 0, 0, 1};
                    cmd_list.Clear(rotated_roi_rtv, clear_clr);

                    const GeneralConstantBuffer* cbs[] = {&rotation_cb};
                    const GpuShaderResourceView* srvs[] = {&delighted_srv};
                    const GpuCommandList::ShaderBinding shader_bindings[] = {
                        {cbs, {}, {}},
                        {{}, srvs, {}},
                    };

                    const GpuRenderTargetView* rtvs[] = {&rotated_roi_rtv};

                    const GpuViewport viewport = {0.0f, 0.0f, static_cast<float>(rotated_size), static_cast<float>(rotated_size)};
                    cmd_list.Render(rotate_pipeline_, {}, nullptr, 4, shader_bindings, rtvs, nullptr, std::span(&viewport, 1), {});
                }

                const uint32_t rotated_width = rotated_roi_tex.Width(0);
                const uint32_t rotated_height = rotated_roi_tex.Height(0);

                GpuTexture2D resized_rotated_roi_x_tex(gpu_system, ResizedImageSize, rotated_height, 1, ColorFmt,
                    GpuResourceFlag::UnorderedAccess, L"resized_rotated_roi_x_tex");
                GpuTexture2D resized_rotated_roi_tex(gpu_system, ResizedImageSize, ResizedImageSize, 1, ColorFmt,
                    GpuResourceFlag::UnorderedAccess, L"resized_rotated_roi_tex");
                {
                    constexpr uint32_t BlockDim = 16;

                    GpuShaderResourceView input_srv(gpu_system, rotated_roi_tex);
                    GpuUnorderedAccessView output_uav(gpu_system, resized_rotated_roi_x_tex);

                    auto downsample_x_cb = ConstantBuffer<ResizeConstantBuffer>(gpu_system, 1, L"downsample_x_cb");
                    downsample_x_cb->src_roi = glm::uvec4(0, 0, rotated_width, rotated_height);
                    downsample_x_cb->dest_size = glm::uvec2(ResizedImageSize, rotated_height);
                    downsample_x_cb->scale = static_cast<float>(rotated_width) / ResizedImageSize;
                    downsample_x_cb->x_dir = true;
                    downsample_x_cb.UploadToGpu();

                    const GeneralConstantBuffer* cbs[] = {&downsample_x_cb};
                    const GpuShaderResourceView* srvs[] = {&input_srv};
                    GpuUnorderedAccessView* uavs[] = {&output_uav};
                    const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                    cmd_list.Compute(
                        resize_pipeline_, DivUp(ResizedImageSize, BlockDim), DivUp(rotated_height, BlockDim), 1, shader_binding);
                }
                {
                    constexpr uint32_t BlockDim = 16;

                    GpuShaderResourceView input_srv(gpu_system, resized_rotated_roi_x_tex);
                    GpuUnorderedAccessView output_uav(gpu_system, resized_rotated_roi_tex);

                    auto downsample_y_cb = ConstantBuffer<ResizeConstantBuffer>(gpu_system, 1, L"downsample_y_cb");
                    downsample_y_cb->src_roi = glm::uvec4(0, 0, ResizedImageSize, rotated_height);
                    downsample_y_cb->dest_size = glm::uvec2(ResizedImageSize, ResizedImageSize);
                    downsample_y_cb->scale = static_cast<float>(rotated_height) / ResizedImageSize;
                    downsample_y_cb->x_dir = false;
                    downsample_y_cb.UploadToGpu();

                    const GeneralConstantBuffer* cbs[] = {&downsample_y_cb};
                    const GpuShaderResourceView* srvs[] = {&input_srv};
                    GpuUnorderedAccessView* uavs[] = {&output_uav};
                    const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                    cmd_list.Compute(
                        resize_pipeline_, DivUp(ResizedImageSize, BlockDim), DivUp(ResizedImageSize, BlockDim), 1, shader_binding);
                }

                auto& resized_rotated_roi_cpu_tex = rotated_images[i];
                resized_rotated_roi_cpu_tex = Texture(
                    resized_rotated_roi_tex.Width(0), resized_rotated_roi_tex.Height(0), ToElementFormat(resized_rotated_roi_tex.Format()));
                resized_rotated_roi_tex.ReadBack(gpu_system, cmd_list, 0, resized_rotated_roi_cpu_tex.Data());
                gpu_system.Execute(std::move(cmd_list));
            }

            return rotated_images;
        }

        Mesh GenMesh(std::span<const Texture> input_images, GpuTexture3D& color_tex)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            GpuCommandList cmd_list;
            uint32_t grid_res;
            GpuTexture3D index_vol_tex;
            GpuBuffer density_features_buff;
            GpuBuffer deformation_features_buff;
            GpuBuffer color_features_buff;

            py_init_future_.wait();

            {
                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();

                auto args = python_system.MakeTuple(4);
                {
                    const uint32_t num_images = static_cast<uint32_t>(input_images.size());
                    auto imgs_args = python_system.MakeTuple(num_images);
                    for (uint32_t i = 0; i < num_images; ++i)
                    {
                        const auto& input_image = input_images[i];
                        auto image_data = python_system.MakeObject(
                            std::span<const std::byte>(reinterpret_cast<const std::byte*>(input_image.Data()), input_image.DataSize()));
                        python_system.SetTupleItem(*imgs_args, i, std::move(image_data));
                    }
                    python_system.SetTupleItem(*args, 0, std::move(imgs_args));

                    python_system.SetTupleItem(*args, 1, python_system.MakeObject(input_images[0].Width()));
                    python_system.SetTupleItem(*args, 2, python_system.MakeObject(input_images[0].Height()));
                    python_system.SetTupleItem(*args, 3, python_system.MakeObject(FormatChannels(input_images[0].Format())));
                }

                python_system.CallObject(*mesh_generator_gen_features_method_, *args);

                const auto py_grid_res = python_system.CallObject(*mesh_generator_resolution_method_);
                grid_res = python_system.Cast<uint32_t>(*py_grid_res);

                const auto py_coords = python_system.CallObject(*mesh_generator_coords_method_);
                const auto coords = python_system.ToSpan<const glm::uvec3>(*py_coords);

                cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Render);

                GpuBuffer coords_buff(gpu_system, static_cast<uint32_t>(coords.size() * sizeof(glm::uvec3)), GpuHeap::Default,
                    GpuResourceFlag::None, L"coords_buff");
                {
                    GpuUploadBuffer coords_upload_buff(gpu_system, coords_buff.Size(), L"coords_upload_buff");
                    std::memcpy(coords_upload_buff.MappedData<glm::uvec3>(), coords.data(), coords_buff.Size());
                    cmd_list.Copy(coords_buff, coords_upload_buff);
                }
                GpuShaderResourceView coords_srv(gpu_system, coords_buff, GpuFormat::RGB32_Uint);

                index_vol_tex = GpuTexture3D(
                    gpu_system, grid_res, grid_res, grid_res, 1, GpuFormat::R32_Uint, GpuResourceFlag::UnorderedAccess, L"index_vol_tex");
                {
                    ConstantBuffer<ScatterIndexConstantBuffer> scatter_index_cb(gpu_system, 1, L"scatter_index_cb");
                    scatter_index_cb->num_features = static_cast<uint32_t>(coords.size());
                    scatter_index_cb.UploadToGpu();

                    GpuUnorderedAccessView index_vol_uav(gpu_system, index_vol_tex);
                    const uint32_t zeros[] = {0, 0, 0, 0};
                    cmd_list.Clear(index_vol_uav, zeros);

                    constexpr uint32_t BlockDim = 256;

                    const GeneralConstantBuffer* cbs[] = {&scatter_index_cb};
                    const GpuShaderResourceView* srvs[] = {&coords_srv};
                    GpuUnorderedAccessView* uavs[] = {&index_vol_uav};
                    const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};

                    cmd_list.Compute(scatter_index_pipeline_, DivUp(static_cast<uint32_t>(coords.size()), BlockDim), 1, 1, shader_binding);
                }

                const auto py_density_features = python_system.CallObject(*mesh_generator_density_features_method_);
                const auto density_features = python_system.ToSpan<const uint16_t>(*py_density_features);

                const auto py_deformation_features = python_system.CallObject(*mesh_generator_deformation_features_method_);
                const auto deformation_features = python_system.ToSpan<const glm::u16vec3>(*py_deformation_features);

                const auto py_color_features = python_system.CallObject(*mesh_generator_color_features_method_);
                const auto color_features = python_system.ToSpan<const glm::u16vec3>(*py_color_features);

                density_features_buff = GpuBuffer(gpu_system, static_cast<uint32_t>(density_features.size() * sizeof(uint16_t)),
                    GpuHeap::Default, GpuResourceFlag::None, L"density_features_buff");
                {
                    GpuUploadBuffer density_features_upload_buff(gpu_system, density_features_buff.Size(), L"density_features_upload_buff");
                    std::memcpy(density_features_upload_buff.MappedData<uint16_t>(), density_features.data(), density_features_buff.Size());
                    cmd_list.Copy(density_features_buff, density_features_upload_buff);
                }

                deformation_features_buff = GpuBuffer(gpu_system, static_cast<uint32_t>(deformation_features.size() * sizeof(glm::u16vec3)),
                    GpuHeap::Default, GpuResourceFlag::None, L"deformation_features_buff");
                {
                    GpuUploadBuffer deformation_features_upload_buff(
                        gpu_system, deformation_features_buff.Size(), L"deformation_features_upload_buff");
                    std::memcpy(deformation_features_upload_buff.MappedData<glm::u16vec3>(), deformation_features.data(),
                        deformation_features_buff.Size());
                    cmd_list.Copy(deformation_features_buff, deformation_features_upload_buff);
                }

                color_features_buff = GpuBuffer(gpu_system, static_cast<uint32_t>(color_features.size() * sizeof(glm::u16vec3)),
                    GpuHeap::Default, GpuResourceFlag::None, L"color_features_buff");
                {
                    GpuUploadBuffer color_features_upload_buff(gpu_system, color_features_buff.Size(), L"color_features_upload_buff");
                    std::memcpy(color_features_upload_buff.MappedData<glm::u16vec3>(), color_features.data(), color_features_buff.Size());
                    cmd_list.Copy(color_features_buff, color_features_upload_buff);
                }
            }

            GpuShaderResourceView density_features_srv(gpu_system, density_features_buff, GpuFormat::R16_Float);
            GpuShaderResourceView deformation_features_srv(gpu_system, deformation_features_buff, GpuFormat::R16_Float);
            GpuShaderResourceView color_features_srv(gpu_system, color_features_buff, GpuFormat::R16_Float);

            const uint32_t size = grid_res + 1;
            GpuTexture3D density_deformation_tex(
                gpu_system, size, size, size, 1, GpuFormat::RGBA16_Float, GpuResourceFlag::UnorderedAccess, L"density_deformation_tex");
            color_tex =
                GpuTexture3D(gpu_system, size, size, size, 1, GpuFormat::RGBA8_UNorm, GpuResourceFlag::UnorderedAccess, L"color_tex");

            {
                ConstantBuffer<GatherVolumeConstantBuffer> gather_volume_cb(gpu_system, 1, L"gather_volume_cb");
                gather_volume_cb->grid_res = grid_res;
                gather_volume_cb->size = size;
                gather_volume_cb.UploadToGpu();

                GpuShaderResourceView index_vol_srv(gpu_system, index_vol_tex);
                GpuUnorderedAccessView density_deformation_uav(gpu_system, density_deformation_tex);
                GpuUnorderedAccessView color_uav(gpu_system, color_tex);

                const float zeros[] = {0, 0, 0, 0};
                cmd_list.Clear(color_uav, zeros);

                constexpr uint32_t BlockDim = 16;

                const GeneralConstantBuffer* cbs[] = {&gather_volume_cb};
                const GpuShaderResourceView* srvs[] = {
                    &index_vol_srv, &density_features_srv, &deformation_features_srv, &color_features_srv};
                GpuUnorderedAccessView* uavs[] = {&density_deformation_uav, &color_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};

                cmd_list.Compute(gather_volume_pipeline_, DivUp(size, BlockDim), DivUp(size, BlockDim), size, shader_binding);
            }

            GpuTexture3D dilated_3d_tmp_gpu_tex(gpu_system, color_tex.Width(0), color_tex.Height(0), color_tex.Depth(0), 1,
                color_tex.Format(), GpuResourceFlag::UnorderedAccess, L"dilated_3d_tmp_gpu_tex");

            GpuTexture3D* dilated_gpu_tex = this->DilateTexture(cmd_list, color_tex, dilated_3d_tmp_gpu_tex);
            if (dilated_gpu_tex != &color_tex)
            {
                color_tex = std::move(*dilated_gpu_tex);
            }

            gpu_system.Execute(std::move(cmd_list));

            Mesh pos_only_mesh = marching_cubes_.Generate(density_deformation_tex, 0, GridScale);
            pos_only_mesh = invisible_faces_remover_.Process(pos_only_mesh);
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
            const auto& vertex_desc = mesh.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);

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

                mesh = mesh.ExtractMesh(vertex_desc, extract_indices);
            }
        }

        Mesh ApplyVertexColor(const Mesh& mesh, const GpuTexture3D& color_vol_tex)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            const uint32_t num_vertices = mesh.NumVertices();
            apply_vertex_color_cb_->num_vertices = num_vertices;
            apply_vertex_color_cb_.UploadToGpu();

            const auto& vertex_desc = mesh.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);

            GpuBuffer pos_vb(
                gpu_system, static_cast<uint32_t>(num_vertices * sizeof(glm::vec3)), GpuHeap::Upload, GpuResourceFlag::None, L"pos_vb");
            {
                glm::vec3* pos_data = reinterpret_cast<glm::vec3*>(pos_vb.Map());
                for (uint32_t i = 0; i < num_vertices; ++i)
                {
                    pos_data[i] = mesh.VertexData<glm::vec3>(i, pos_attrib_index);
                }
                pos_vb.Unmap(GpuRange{0, pos_vb.Size()});
            }

            GpuShaderResourceView pos_srv(gpu_system, pos_vb, GpuFormat::RGB32_Float);
            GpuShaderResourceView color_vol_srv(gpu_system, color_vol_tex);

            GpuBuffer color_vb(gpu_system, static_cast<uint32_t>(num_vertices * sizeof(glm::vec3)), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, L"pos_color_vb");
            GpuUnorderedAccessView color_uav(gpu_system, color_vb, GpuFormat::R32_Float);

            constexpr uint32_t BlockDim = 256;

            GpuCommandList cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Render);

            const GeneralConstantBuffer* cbs[] = {&apply_vertex_color_cb_};
            const GpuShaderResourceView* srvs[] = {&color_vol_srv, &pos_srv};
            GpuUnorderedAccessView* uavs[] = {&color_uav};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(apply_vertex_color_pipeline_, DivUp(num_vertices, BlockDim), 1, 1, shader_binding);

            GpuReadBackBuffer color_read_back_vb(
                gpu_system, static_cast<uint32_t>(num_vertices * sizeof(glm::vec3)), L"color_read_back_vb");
            cmd_list.Copy(color_read_back_vb, color_vb);

            gpu_system.Execute(std::move(cmd_list));
            gpu_system.CpuWait();

            const VertexAttrib pos_color_vertex_attribs[] = {
                {VertexAttrib::Semantic::Position, 0, 3},
                {VertexAttrib::Semantic::Color, 0, 3},
            };
            constexpr uint32_t OutputPosAttribIndex = 0;
            constexpr uint32_t OutputColorAttribIndex = 1;

            Mesh pos_color_mesh(VertexDesc(pos_color_vertex_attribs), mesh.NumVertices(), static_cast<uint32_t>(mesh.IndexBuffer().size()));
            const auto* colors = color_read_back_vb.MappedData<glm::vec3>();
            for (uint32_t i = 0; i < num_vertices; ++i)
            {
                pos_color_mesh.VertexData<glm::vec3>(i, OutputPosAttribIndex) = mesh.VertexData<glm::vec3>(i, pos_attrib_index);
                pos_color_mesh.VertexData<glm::vec3>(i, OutputColorAttribIndex) = colors[i];
            }

            pos_color_mesh.IndexBuffer(mesh.IndexBuffer());

            return pos_color_mesh;
        }

        void FillHoles(Mesh& mesh)
        {
            std::vector<std::vector<uint32_t>> holes = this->FindHoles(mesh);
            std::vector<uint32_t> new_indices;
            for (auto& hole : holes)
            {
                this->TriangulateHole(mesh, hole, new_indices);
            }

            if (!new_indices.empty())
            {
                const uint32_t base = static_cast<uint32_t>(mesh.IndexBuffer().size());
                mesh.ResizeIndices(base + static_cast<uint32_t>(new_indices.size()));
                std::memcpy(&mesh.IndexBuffer()[base], new_indices.data(), new_indices.size() * sizeof(new_indices[0]));
            }
        }

        std::vector<std::vector<uint32_t>> FindHoles(const Mesh& mesh) const
        {
            const uint32_t num_indices = static_cast<uint32_t>(mesh.IndexBuffer().size());
            std::map<std::tuple<uint32_t, uint32_t>, uint32_t> edge_count;
            for (uint32_t i = 0; i < num_indices; i += 3)
            {
                const uint32_t v0 = mesh.Index(i + 0);
                const uint32_t v1 = mesh.Index(i + 1);
                const uint32_t v2 = mesh.Index(i + 2);

                auto add_edge = [&edge_count](uint32_t lhs, uint32_t rhs) {
                    if (lhs > rhs)
                    {
                        std::swap(lhs, rhs);
                    }
                    ++edge_count[{lhs, rhs}];
                };

                add_edge(v0, v1);
                add_edge(v1, v2);
                add_edge(v2, v0);
            }

            std::vector<std::tuple<uint32_t, uint32_t>> boundary_edges;
            for (const auto& [edge, count] : edge_count)
            {
                if (count == 1)
                {
                    boundary_edges.push_back(edge);
                }
            }

            std::vector<std::vector<uint32_t>> holes;
            while (!boundary_edges.empty())
            {
                const std::tuple<uint32_t, uint32_t> start_edge = boundary_edges.back();
                boundary_edges.pop_back();
                std::vector<uint32_t> loop = {std::get<0>(start_edge), std::get<1>(start_edge)};

                for (;;)
                {
                    bool found_next_edge = false;
                    for (auto iter = boundary_edges.begin(); iter != boundary_edges.end(); ++iter)
                    {
                        const bool at_beg = std::get<0>(*iter) == loop.back();
                        const bool at_end = std::get<1>(*iter) == loop.back();
                        if (at_beg || at_end)
                        {
                            loop.push_back(at_beg ? std::get<1>(*iter) : std::get<0>(*iter));
                            boundary_edges.erase(iter);
                            found_next_edge = true;
                            break;
                        }
                    }

                    if (!found_next_edge)
                    {
                        break;
                    }
                }

                if (loop.size() >= 2)
                {
                    if (loop.front() == loop.back())
                    {
                        loop.pop_back();
                    }

                    holes.emplace_back(std::move(loop));
                }
            }

            return holes;
        }

        bool IsEar(const Mesh& mesh, const std::vector<uint32_t>& hole, size_t index)
        {
            const auto& vertex_desc = mesh.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);

            const size_t num = hole.size();
            const size_t prev = (index == 0) ? num - 1 : index - 1;
            const size_t next = (index + 1) % num;
            const glm::vec3& p0 = mesh.VertexData<glm::vec3>(hole[prev], pos_attrib_index);
            const glm::vec3& p1 = mesh.VertexData<glm::vec3>(hole[index], pos_attrib_index);
            const glm::vec3& p2 = mesh.VertexData<glm::vec3>(hole[next], pos_attrib_index);

            const glm::vec3 edge1 = p1 - p0;
            const glm::vec3 edge2 = p2 - p0;
            const glm::vec3 normal = glm::cross(edge1, edge2);
            const float area = normal.z;
            if (area <= 0)
            {
                return false;
            }

            for (size_t i = 0; i < num; ++i)
            {
                if ((i != prev) && (i != index) && (i != next))
                {
                    const glm::vec3& p = mesh.VertexData<glm::vec3>(hole[i], pos_attrib_index);
                    const glm::vec3 w = p - p0;

                    // Compute barycentric coordinates
                    const float uu = glm::dot(edge1, edge1);
                    const float uv = glm::dot(edge1, edge2);
                    const float vv = glm::dot(edge2, edge2);
                    const float wu = glm::dot(w, edge1);
                    const float wv = glm::dot(w, edge2);

                    // Denominator for barycentric coordinates
                    const float denom = uv * uv - uu * vv;

                    // Barycentric coordinates beta and gamma
                    const float beta = (uv * wv - vv * wu) / denom;
                    const float gamma = (uv * wu - uu * wv) / denom;
                    const float alpha = 1.0f - beta - gamma;

                    // Check if point is inside the triangle
                    // Allow small epsilon for floating-point errors
                    constexpr float Epsilon = 1e-6f;
                    if ((alpha >= -Epsilon) && (beta >= -Epsilon) && (gamma >= -Epsilon) && (alpha <= 1.0f + Epsilon) &&
                        (beta <= 1.0f + Epsilon) && (gamma <= 1.0f + Epsilon) && (std::abs(alpha + beta + gamma - 1.0f) < Epsilon))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        void TriangulateHole(const Mesh& mesh, std::vector<uint32_t>& hole, std::vector<uint32_t>& new_indices)
        {
            const auto& vertex_desc = mesh.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);

            size_t num = hole.size();
            if (num < 3)
            {
                return;
            }

            glm::vec3 centroid(0.0f);
            for (const uint32_t idx : hole)
            {
                centroid += mesh.VertexData<glm::vec3>(idx, pos_attrib_index);
            }
            centroid /= static_cast<float>(num);

            // Compute the average normal (Newell's method)
            glm::vec3 plane_normal(0.0f);
            for (size_t i = 0; i < num; ++i)
            {
                const glm::vec3& v0 = mesh.VertexData<glm::vec3>(hole[i], pos_attrib_index);
                const glm::vec3& v1 = mesh.VertexData<glm::vec3>(hole[(i + 1) % num], pos_attrib_index);
                plane_normal.x += (v0.y - v1.y) * (v0.z + v1.z);
                plane_normal.y += (v0.z - v1.z) * (v0.x + v1.x);
                plane_normal.z += (v0.x - v1.x) * (v0.y + v1.y);
            }
            if (glm::dot(plane_normal, plane_normal) < 1e-6f)
            {
                // Invalid plane, skip triangulation
                return;
            }
            plane_normal = glm::normalize(plane_normal);

            glm::vec3 u = mesh.VertexData<glm::vec3>(hole[1], pos_attrib_index) - mesh.VertexData<glm::vec3>(hole[0], pos_attrib_index);
            if (glm::dot(u, u) < 1e-6f)
            {
                return;
            }
            const glm::vec3 v = glm::normalize(glm::cross(plane_normal, u));
            u = glm::cross(v, plane_normal);

            std::vector<glm::vec2> projected(num);
            for (size_t i = 0; i < num; ++i)
            {
                const glm::vec3 vec = mesh.VertexData<glm::vec3>(hole[i], pos_attrib_index) - centroid;
                projected[i] = glm::vec2(glm::dot(vec, u), glm::dot(vec, v));
            }

            float signed_area = 0;
            for (size_t i = 0; i < num; ++i)
            {
                const size_t next = (i + 1) % num;
                signed_area += projected[i].x * projected[next].y - projected[next].x * projected[i].y;
            }

            // If area is negative, reverse the hole (clockwise to counterclockwise)
            if (signed_area < 0)
            {
                std::reverse(hole.begin(), hole.end());
            }

            while (hole.size() > 3)
            {
                num = hole.size();
                bool ear_found = false;
                for (size_t i = 0; i < num; i++)
                {
                    if (this->IsEar(mesh, hole, i))
                    {
                        const size_t prev = (i == 0) ? num - 1 : i - 1;
                        const size_t next = (i + 1) % num;
                        new_indices.push_back(hole[prev]);
                        new_indices.push_back(hole[i]);
                        new_indices.push_back(hole[next]);
                        hole.erase(hole.begin() + i);
                        ear_found = true;
                        break;
                    }
                }
                if (!ear_found)
                {
                    break;
                }
            }

            if (hole.size() == 3)
            {
                new_indices.push_back(hole[0]);
                new_indices.push_back(hole[1]);
                new_indices.push_back(hole[2]);
            }

            for (size_t i = 0; i < new_indices.size(); i += 3)
            {
                const glm::vec3& p0 = mesh.VertexData<glm::vec3>(new_indices[i + 0], pos_attrib_index);
                const glm::vec3& p1 = mesh.VertexData<glm::vec3>(new_indices[i + 1], pos_attrib_index);
                const glm::vec3& p2 = mesh.VertexData<glm::vec3>(new_indices[i + 2], pos_attrib_index);

                const glm::vec3 edge1 = p1 - p0;
                const glm::vec3 edge2 = p2 - p0;
                const glm::vec3 normal = glm::cross(edge1, edge2);
                if (glm::dot(normal, plane_normal) < 0)
                {
                    std::swap(new_indices[i + 1], new_indices[i + 2]);
                }
            }
        }

        void MergeTexture(GpuCommandList& cmd_list, const GpuTexture3D& color_vol_tex, const GpuTexture2D& pos_tex,
            const glm::mat4x4& model_mtx, GpuTexture2D& color_tex)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            GpuUnorderedAccessView merged_uav(gpu_system, color_tex);

            const uint32_t texture_size = color_tex.Width(0);
            merge_texture_cb_->inv_model = glm::transpose(glm::inverse(model_mtx));
            merge_texture_cb_->texture_size = texture_size;
            merge_texture_cb_.UploadToGpu();

            GpuShaderResourceView pos_srv(gpu_system, pos_tex);
            GpuShaderResourceView color_vol_srv(gpu_system, color_vol_tex);

            constexpr uint32_t BlockDim = 16;

            const GeneralConstantBuffer* cbs[] = {&merge_texture_cb_};
            const GpuShaderResourceView* srvs[] = {&color_vol_srv, &pos_srv};
            GpuUnorderedAccessView* uavs[] = {&merged_uav};
            const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
            cmd_list.Compute(merge_texture_pipeline_, DivUp(texture_size, BlockDim), DivUp(texture_size, BlockDim), 1, shader_binding);
        }

        glm::mat4x4 GuessModelMatrix(const StructureFromMotion::Result& sfm_input, const Obb& obb, const glm::vec3& centroid,
            const glm::vec3& local_up_vec, const glm::vec3& up_vec)
        {
            const glm::mat4 init_model_mtx =
                glm::translate(glm::identity<glm::mat4x4>(), centroid) * glm::mat4_cast(glm::rotation(local_up_vec, up_vec));

            glm::vec3 corners[8];
            Obb::GetCorners(obb, corners);

            float scale = 1e10f;
            for (uint32_t i = 0; i < sfm_input.views.size(); ++i)
            {
                const auto& view = sfm_input.views[i];
                const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                const glm::mat4x4 view_mtx = CalcViewMatrix(view);
                const glm::mat4x4 proj_mtx = CalcProjMatrix(intrinsic, 0.1f, 30.0f);
                const glm::mat4x4 mvp_mtx = proj_mtx * view_mtx * init_model_mtx;

                constexpr float RoiScale = 1.6f;
                const float roi_fx = RoiScale * 2.0f * view.delighted_image.Width() / intrinsic.width;
                const float roi_fy = RoiScale * 2.0f * view.delighted_image.Height() / intrinsic.height;
                const float roi_fz = RoiScale;

                glm::vec3 aabb_min_ps(+1e10f, +1e10f, +1e10f);
                glm::vec3 aabb_max_ps(-1e10f, -1e10f, -1e10f);
                for (const auto& corner : corners)
                {
                    const glm::vec4 pos_ps = mvp_mtx * glm::vec4(corner - obb.center, 1);
                    const glm::vec3 pos_ps3 = glm::vec3(pos_ps) / pos_ps.w;

                    aabb_min_ps = glm::min(aabb_min_ps, pos_ps3);
                    aabb_max_ps = glm::max(aabb_max_ps, pos_ps3);
                }

                const glm::vec3 aabb_extents = aabb_max_ps - aabb_min_ps;
                scale = std::min({scale, roi_fx / aabb_extents.x, roi_fy / aabb_extents.y, roi_fz / aabb_extents.z});
            }

            return glm::scale(init_model_mtx, glm::vec3(scale));
        }

        template <typename GpuTextureT>
        GpuTextureT* DilateTexture(GpuCommandList& cmd_list, GpuTextureT& tex, GpuTextureT& tmp_tex)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            constexpr uint32_t BlockDim = 16;
            constexpr uint32_t DilateTimes = std::is_same_v<GpuTextureT, GpuTexture2D> ? Dilate2DTimes : Dilate3DTimes;
            const GpuComputePipeline& dilate_pipeline = std::is_same_v<GpuTextureT, GpuTexture2D> ? dilate_pipeline_ : dilate_3d_pipeline_;

            ConstantBuffer<DilateConstantBuffer> dilate_cb(gpu_system, 1, L"dilate_cb");
            dilate_cb->texture_size = tex.Width(0);
            dilate_cb.UploadToGpu();

            GpuShaderResourceView tex_srv(gpu_system, tex);
            GpuShaderResourceView tmp_tex_srv(gpu_system, tmp_tex);
            GpuUnorderedAccessView tex_uav(gpu_system, tex);
            GpuUnorderedAccessView tmp_tex_uav(gpu_system, tmp_tex);

            GpuTextureT* texs[] = {&tex, &tmp_tex};
            GpuShaderResourceView* tex_srvs[] = {&tex_srv, &tmp_tex_srv};
            GpuUnorderedAccessView* tex_uavs[] = {&tex_uav, &tmp_tex_uav};
            for (uint32_t i = 0; i < DilateTimes; ++i)
            {
                const uint32_t src = i & 1;
                const uint32_t dst = src ? 0 : 1;

                const GeneralConstantBuffer* cbs[] = {&dilate_cb};
                const GpuShaderResourceView* srvs[] = {tex_srvs[src]};
                GpuUnorderedAccessView* uavs[] = {tex_uavs[dst]};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(dilate_pipeline, DivUp(texs[dst]->Width(0), BlockDim), DivUp(texs[dst]->Height(0), BlockDim),
                    texs[dst]->Depth(0), shader_binding);
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
        AIHoloImagerInternal& aihi_;

        PyObjectPtr mesh_generator_module_;
        PyObjectPtr mesh_generator_class_;
        PyObjectPtr mesh_generator_;
        PyObjectPtr mesh_generator_gen_features_method_;
        PyObjectPtr mesh_generator_resolution_method_;
        PyObjectPtr mesh_generator_coords_method_;
        PyObjectPtr mesh_generator_density_features_method_;
        PyObjectPtr mesh_generator_deformation_features_method_;
        PyObjectPtr mesh_generator_color_features_method_;
        std::future<void> py_init_future_;

        InvisibleFacesRemover invisible_faces_remover_;
        MarchingCubes marching_cubes_;
        TextureReconstruction texture_recon_;

        struct RotateConstantBuffer
        {
            glm::mat4x4 rotation_mtx;
            glm::vec4 tc_bounding_box;
        };
        GpuRenderPipeline rotate_pipeline_;

        struct ResizeConstantBuffer
        {
            glm::uvec4 src_roi;
            glm::uvec2 dest_size;
            float scale;
            uint32_t x_dir;
        };
        GpuComputePipeline resize_pipeline_;

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
        GpuComputePipeline dilate_pipeline_;
        GpuComputePipeline dilate_3d_pipeline_;

        struct ApplyVertexColorConstantBuffer
        {
            uint32_t num_vertices;
            float inv_scale;
            uint32_t padding[2];
        };
        ConstantBuffer<ApplyVertexColorConstantBuffer> apply_vertex_color_cb_;
        GpuComputePipeline apply_vertex_color_pipeline_;

        static constexpr uint32_t Dilate2DTimes = 4;
        static constexpr uint32_t Dilate3DTimes = 8;
        static constexpr float GridScale = 2.0f;
        static constexpr uint32_t ResizedImageSize = 518;

        static constexpr GpuFormat ColorFmt = GpuFormat::RGBA8_UNorm;
    };

    MeshGenerator::MeshGenerator(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    MeshGenerator::~MeshGenerator() noexcept = default;

    MeshGenerator::MeshGenerator(MeshGenerator&& other) noexcept = default;
    MeshGenerator& MeshGenerator::operator=(MeshGenerator&& other) noexcept = default;

    Mesh MeshGenerator::Generate(const StructureFromMotion::Result& sfm_input, uint32_t texture_size, const std::filesystem::path& tmp_dir)
    {
        return impl_->Generate(sfm_input, texture_size, tmp_dir);
    }
} // namespace AIHoloImager
