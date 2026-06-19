// Copyright (c) 2024-2026 Minmin Gong
//

#include "MeshGenerator.hpp"

#include <array>
#include <cassert>
#include <future>
#include <iostream>
#include <set>
#include <tuple>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 5054) // Ignore operator between enums of different types
#endif
#include <Eigen/Core>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif
#include <Eigen/Eigenvalues>
#include <glm/gtc/quaternion.hpp>
#ifndef GLM_ENABLE_EXPERIMENTAL
    #define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/quaternion.hpp>
#include <glm/vec3.hpp>

#include "Base/Util.hpp"
#include "GSplat/GaussianSplatting.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuSampler.hpp"
#include "Gpu/GpuTexture.hpp"
#include "InvisibleFacesRemover.hpp"
#include "MarchingCubes.hpp"
#include "Util/BoundingBox.hpp"
#include "Util/PerfProfiler.hpp"

#include "CompiledShader/MeshGen/ApplyVertexColorCs.h"
#include "CompiledShader/MeshGen/Dilate3DCs.h"
#include "CompiledShader/MeshGen/ErosionDilationMaskCs.h"
#include "CompiledShader/MeshGen/GatherVolumeCs.h"
#include "CompiledShader/MeshGen/ResizeCs.h"
#include "CompiledShader/MeshGen/RotatePs.h"
#include "CompiledShader/MeshGen/RotateVs.h"
#include "CompiledShader/MeshGen/ScatterIndexCs.h"

namespace AIHoloImager
{
    class MeshGenerator::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi), invisible_faces_remover_(aihi), marching_cubes_(aihi)
        {
            PerfRegion init_perf(aihi_.PerfProfilerInstance(), "Mesh generator init");

            py_init_future_ = std::async(std::launch::async, [this] {
                PerfRegion init_async_perf(aihi_.PerfProfilerInstance(), "Mesh generator init (async)");

                PythonSystem::GilGuard guard;

                auto& gpu_system = aihi_.GpuSystemInstance();
                auto& python_system = aihi_.PythonSystemInstance();

                mesh_generator_module_ = python_system.Import("MeshGenerator");
                mesh_generator_class_ = python_system.GetAttr(*mesh_generator_module_, "MeshGenerator");
                mesh_generator_ = python_system.CallObject(*mesh_generator_class_, reinterpret_cast<void*>(&gpu_system));
                mesh_generator_gen_features_method_ = python_system.GetAttr(*mesh_generator_, "GenFeatures");
                mesh_generator_resolution_method_ = python_system.GetAttr(*mesh_generator_, "Resolution");
                mesh_generator_coords_method_ = python_system.GetAttr(*mesh_generator_, "Coords");
                mesh_generator_density_features_method_ = python_system.GetAttr(*mesh_generator_, "DensityFeatures");
                mesh_generator_deformation_features_method_ = python_system.GetAttr(*mesh_generator_, "DeformationFeatures");
                mesh_generator_color_features_method_ = python_system.GetAttr(*mesh_generator_, "ColorFeatures");
                mesh_generator_gsplat_num_gaussians_method_ = python_system.GetAttr(*mesh_generator_, "GSplatNumGaussians");
                mesh_generator_gsplat_sh_coefficients_method_ = python_system.GetAttr(*mesh_generator_, "GSplatShCoefficients");
                mesh_generator_gsplat_positions_method_ = python_system.GetAttr(*mesh_generator_, "GSplatPositions");
                mesh_generator_gsplat_scales_method_ = python_system.GetAttr(*mesh_generator_, "GSplatScales");
                mesh_generator_gsplat_rotations_method_ = python_system.GetAttr(*mesh_generator_, "GSplatRotations");
                mesh_generator_gsplat_shs_method_ = python_system.GetAttr(*mesh_generator_, "GSplatShs");
                mesh_generator_gsplat_opacities_method_ = python_system.GetAttr(*mesh_generator_, "GSplatOpacities");
            });

            auto& gpu_system = aihi_.GpuSystemInstance();

            {
                const ShaderInfo shader = {DEFINE_SHADER(ErosionDilationMaskCs)};
                erosion_dilation_mask_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shaders[] = {
                    {DEFINE_SHADER(RotateVs)},
                    {DEFINE_SHADER(RotatePs)},
                };

                const GpuFormat rtv_formats[] = {ColorFmt};
                const GpuRenderPipeline::States states{
                    .cull_mode = GpuRenderPipeline::CullMode::CounterClockWise,
                    .rtv_formats = rtv_formats,
                };

                const GpuStaticSampler bilinear_sampler(gpu_system, {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear},
                    GpuStaticSampler::AddressMode::Border);

                rotate_pipeline_ = GpuRenderPipeline(gpu_system, GpuRenderPipeline::PrimitiveTopology::TriangleStrip, shaders,
                    GpuVertexLayout(gpu_system, {}), std::span(&bilinear_sampler, 1), states);
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(ResizeCs)};
                resize_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(ScatterIndexCs)};
                scatter_index_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(GatherVolumeCs)};
                gather_volume_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }

            const GpuStaticSampler trilinear_sampler(
                gpu_system, {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear}, GpuStaticSampler::AddressMode::Clamp);
            {
                const ShaderInfo shader = {DEFINE_SHADER(Dilate3DCs)};
                dilate_3d_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(ApplyVertexColorCs)};
                apply_vertex_color_pipeline_ = GpuComputePipeline(gpu_system, shader, std::span{&trilinear_sampler, 1});
            }
        }

        ~Impl()
        {
            PerfRegion destroy_perf(aihi_.PerfProfilerInstance(), "MeshGenerator destroy");

            if (!py_init_finished_)
            {
                py_init_future_.wait();
            }

            PythonSystem::GilGuard guard;

            auto& python_system = aihi_.PythonSystemInstance();
            auto mesh_generator_destroy_method = python_system.GetAttr(*mesh_generator_, "Destroy");
            python_system.CallObject(*mesh_generator_destroy_method);

            mesh_generator_destroy_method.reset();
            mesh_generator_gsplat_num_gaussians_method_.reset();
            mesh_generator_gsplat_sh_coefficients_method_.reset();
            mesh_generator_gsplat_positions_method_.reset();
            mesh_generator_gsplat_scales_method_.reset();
            mesh_generator_gsplat_rotations_method_.reset();
            mesh_generator_gsplat_shs_method_.reset();
            mesh_generator_gsplat_opacities_method_.reset();
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

        Result Generate(const StructureFromMotion::Result& sfm_input)
        {
#ifdef AIHI_KEEP_INTERMEDIATES
            auto& gpu_system = aihi_.GpuSystemInstance();
#endif
            auto& profiler = aihi_.PerfProfilerInstance();
            PerfRegion process_perf(profiler, "Mesh generator process");

            const auto output_dir = aihi_.TmpDir() / "MeshGen";
            std::filesystem::create_directories(output_dir);

            Aabb obj_aabb;
            glm::vec3 up_vec;
            std::vector<GpuTexture2D> rotated_images;
            {
                std::cout << "Rotating images...\n";

                PerfRegion rotating_perf(profiler, "Rotating images");

                this->StatForegroundObject(obj_aabb, up_vec, sfm_input, output_dir);
                rotated_images = this->RotateImages(sfm_input, up_vec);

#ifdef AIHI_KEEP_INTERMEDIATES
                {
                    glm::vec3 corners[8];
                    Aabb::GetCorners(obj_aabb, corners);

                    const Mesh bb_mesh = BoxMesh(corners);
                    SaveMesh(bb_mesh, output_dir / "Aabb.glb");
                }

                for (size_t i = 0; i < rotated_images.size(); ++i)
                {
                    auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);
                    Texture rotated_image(rotated_images[i].Width(0), rotated_images[i].Height(0), ElementFormat::RGBA8_UNorm);
                    const auto rb_future = cmd_list.ReadBackAsync(rotated_images[i], 0, rotated_image.Data(), rotated_image.DataSize());
                    gpu_system.Execute(std::move(cmd_list));
                    rb_future.wait();

                    SaveTexture(rotated_image, output_dir / std::format("Rotated_{}.png", i));
                }
#endif
            }

            GpuMesh mesh;
            Gaussians gaussians;
            {
                std::cout << "Generating mesh from images...\n";

                PerfRegion gen_mesh_perf(profiler, "Generate mesh");

                GpuTexture3D color_vol_tex;
                mesh = this->GenMesh(rotated_images, color_vol_tex, gaussians);

#ifdef AIHI_KEEP_INTERMEDIATES
                SaveMesh(ToMesh(gpu_system, mesh), output_dir / "AiMeshPosOnly.glb");
                SavePointCloud(gpu_system, gaussians, output_dir / "AiMeshGaussians.ply");
#endif

                mesh = this->ApplyVertexColor(mesh, color_vol_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
                SaveMesh(ToMesh(gpu_system, mesh), output_dir / "AiMeshPosColor.glb");
#endif
            }

            return {std::move(mesh), std::move(gaussians), std::move(obj_aabb), std::move(up_vec)};
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

        void StatForegroundObject(Aabb& bb, glm::vec3& up_vec, const StructureFromMotion::Result& sfm_input,
            [[maybe_unused]] const std::filesystem::path& tmp_dir)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            std::vector<uint8_t> point_used(sfm_input.structure.size(), 0);
            std::vector<glm::vec3> object_points;
            std::vector<glm::vec3> plane_points;
#ifdef AIHI_KEEP_INTERMEDIATES
            std::vector<glm::vec3> object_colors;
            std::vector<glm::vec3> plane_colors;
#endif
            for (uint32_t i = 0; i < sfm_input.projections.size(); ++i)
            {
                const auto& projection = sfm_input.projections[i];

                const uint32_t delighted_width = projection.image->Width(0);
                const uint32_t delighted_height = projection.image->Height(0);

#ifdef AIHI_KEEP_INTERMEDIATES
                Texture delighted_image(delighted_width, delighted_height, ElementFormat::RGBA8_UNorm);
                const uint32_t delighted_fmt_size = FormatChannels(delighted_image.Format());
#endif
                Texture erosion_mask_image(delighted_width, delighted_height, ElementFormat::R8_UNorm);
                Texture dilation_mask_image(delighted_width, delighted_height, ElementFormat::R8_UNorm);
                {
                    auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);

                    std::future<void> erosion_rb_future;
                    {
                        constexpr uint32_t BlockDim = 16;

                        GpuTexture2D erosion_mask_gpu_texs[2];
                        for (auto& tex : erosion_mask_gpu_texs)
                        {
                            tex = GpuTexture2D(gpu_system, delighted_width, delighted_height, 1, GpuFormat::R8_UNorm,
                                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "erosion_mask_gpu_tex");
                        }

                        uint32_t dst;
                        for (size_t j = 0; j < 4; ++j)
                        {
                            const uint32_t src = j & 1;
                            dst = src ? 0 : 1;

                            GpuConstantBufferOfType<ErosionMaskConstantBuffer> erosion_mask_cb(gpu_system, "erosion_mask_cb");
                            erosion_mask_cb->texture_size = {delighted_width, delighted_height};
                            erosion_mask_cb->erosion = true;
                            erosion_mask_cb->channel = j == 0 ? 3 : 0;
                            erosion_mask_cb.UploadStaging();
                            const GpuConstantBufferView erosion_cbv(gpu_system, erosion_mask_cb);

                            const GpuShaderResourceView input_srv(gpu_system, j == 0 ? *projection.image : erosion_mask_gpu_texs[src]);
                            GpuUnorderedAccessView erosion_uav(gpu_system, erosion_mask_gpu_texs[dst]);

                            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                                {"param_cb", &erosion_cbv},
                            };
                            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                                {"input_tex", &input_srv},
                            };
                            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                                {"output_tex", &erosion_uav},
                            };
                            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                            cmd_list.Compute(erosion_dilation_mask_pipeline_,
                                {DivUp(delighted_width, BlockDim), DivUp(delighted_height, BlockDim), 1}, shader_binding);
                        }

                        erosion_rb_future =
                            cmd_list.ReadBackAsync(erosion_mask_gpu_texs[dst], 0, erosion_mask_image.Data(), erosion_mask_image.DataSize());
                    }

                    std::future<void> dilation_rb_future;
                    {
                        constexpr uint32_t BlockDim = 16;

                        GpuTexture2D dilation_mask_gpu_texs[2];
                        for (auto& tex : dilation_mask_gpu_texs)
                        {
                            tex = GpuTexture2D(gpu_system, delighted_width, delighted_height, 1, GpuFormat::R8_UNorm,
                                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "dilation_mask_gpu_texs");
                        }

                        uint32_t dst;
                        for (size_t j = 0; j < 1; ++j)
                        {
                            const uint32_t src = j & 1;
                            dst = src ? 0 : 1;

                            GpuConstantBufferOfType<ErosionMaskConstantBuffer> dilation_mask_cb(gpu_system, "dilation_mask_cb");
                            dilation_mask_cb->texture_size = {delighted_width, delighted_height};
                            dilation_mask_cb->erosion = false;
                            dilation_mask_cb->channel = j == 0 ? 3 : 0;
                            dilation_mask_cb.UploadStaging();
                            const GpuConstantBufferView dilation_cbv(gpu_system, dilation_mask_cb);

                            const GpuShaderResourceView input_srv(gpu_system, j == 0 ? *projection.image : dilation_mask_gpu_texs[src]);
                            GpuUnorderedAccessView dilation_uav(gpu_system, dilation_mask_gpu_texs[dst]);

                            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                                {"param_cb", &dilation_cbv},
                            };
                            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                                {"input_tex", &input_srv},
                            };
                            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                                {"output_tex", &dilation_uav},
                            };
                            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                            cmd_list.Compute(erosion_dilation_mask_pipeline_,
                                {DivUp(delighted_width, BlockDim), DivUp(delighted_height, BlockDim), 1}, shader_binding);
                        }

                        dilation_rb_future = cmd_list.ReadBackAsync(
                            dilation_mask_gpu_texs[dst], 0, dilation_mask_image.Data(), dilation_mask_image.DataSize());
                    }

#ifdef AIHI_KEEP_INTERMEDIATES
                    std::future<void> delighted_rb_future =
                        cmd_list.ReadBackAsync(*projection.image, 0, delighted_image.Data(), delighted_image.DataSize());
#endif

                    gpu_system.Execute(std::move(cmd_list));

                    erosion_rb_future.wait();
                    dilation_rb_future.wait();
#ifdef AIHI_KEEP_INTERMEDIATES
                    delighted_rb_future.wait();
#endif
                }

                const std::byte* erosion_mask_image_data = erosion_mask_image.Data();
                const std::byte* dilation_mask_image_data = dilation_mask_image.Data();
#ifdef AIHI_KEEP_INTERMEDIATES
                const std::byte* delighted_image_data = delighted_image.Data();
#endif

                constexpr uint32_t Gap = 32;
                const uint32_t beg_x =
                    static_cast<uint32_t>(std::max(static_cast<int32_t>(projection.image_offset.x) - static_cast<int32_t>(Gap), 0));
                const uint32_t beg_y = projection.image_offset.y + delighted_height - Gap;
                const uint32_t end_x = projection.image_offset.x + delighted_width + Gap;
                const uint32_t end_y = projection.image_offset.y + delighted_height + Gap;

                const uint32_t delighted_beg_x = projection.image_offset.x;
                const uint32_t delighted_beg_y = projection.image_offset.y;
                const uint32_t delighted_end_x = projection.image_offset.x + delighted_width;
                const uint32_t delighted_end_y = projection.image_offset.y + delighted_height;

                for (size_t j = 0; j < sfm_input.structure.size(); ++j)
                {
                    const auto& landmark = sfm_input.structure[j];
                    for (const auto& ob : landmark.obs)
                    {
                        if (ob.view_id == i)
                        {
                            const uint32_t x = static_cast<uint32_t>(std::round(ob.point.x));
                            const uint32_t y = static_cast<uint32_t>(std::round(ob.point.y));

                            const uint32_t delighted_x = x - delighted_beg_x;
                            const uint32_t delighted_y = y - delighted_beg_y;

                            const glm::vec3 point = landmark.point;
#ifdef AIHI_KEEP_INTERMEDIATES
                            glm::vec3 color(0, 0, 0);
                            if ((x >= delighted_beg_x) && (y >= delighted_beg_y) && (x < delighted_end_x) && (y < delighted_end_y))
                            {
                                const uint32_t offset = (delighted_y * delighted_width + delighted_x) * delighted_fmt_size;
                                const float r = static_cast<int>(delighted_image_data[offset + 0]) + 0.5f;
                                const float g = static_cast<int>(delighted_image_data[offset + 1]) + 0.5f;
                                const float b = static_cast<int>(delighted_image_data[offset + 2]) + 0.5f;
                                color = glm::vec3(r, g, b) / 255.0f;
                            }
#endif

                            if (!(point_used[j] & 0x1U) && (x >= delighted_beg_x) && (y >= delighted_beg_y) && (x < delighted_end_x) &&
                                (y < delighted_end_y))
                            {
                                if ((delighted_x < delighted_width) && (delighted_y < delighted_height))
                                {
                                    const std::byte mask = erosion_mask_image_data[delighted_y * delighted_width + delighted_x];
                                    if (mask > std::byte(0x7F))
                                    {
                                        object_points.push_back(point);
#ifdef AIHI_KEEP_INTERMEDIATES
                                        object_colors.push_back(color);
#endif
                                        point_used[j] |= 0x1U;
                                    }
                                }
                            }

                            if (!(point_used[j] & 0x2U) && (x >= beg_x) && (y >= beg_y) && (x < end_x) && (y < end_y))
                            {
                                if ((delighted_x < delighted_width) && (delighted_y < delighted_height))
                                {
                                    const std::byte mask = dilation_mask_image_data[delighted_y * delighted_width + delighted_x];
                                    if (mask <= std::byte(0x7F))
                                    {
                                        plane_points.push_back(point);
#ifdef AIHI_KEEP_INTERMEDIATES
                                        plane_colors.push_back(color);
#endif
                                        point_used[j] |= 0x2U;
                                    }
                                }
                            }

                            break;
                        }
                    }
                }
            }

            glm::vec3 object_center(0, 0, 0);
            for (size_t i = 0; i < object_points.size(); ++i)
            {
                object_center += object_points[i];
            }
            object_center /= static_cast<float>(object_points.size());

            glm::vec4 plane = this->FitPlane(plane_points);
            if (glm::dot(glm::vec3(plane), object_center) + plane.w < 0)
            {
                plane = -plane;
            }

            up_vec = glm::vec3(plane);

            constexpr float GroundThreshold = 0.01f;

            bb = Aabb();
            for (size_t i = 0; i < object_points.size(); ++i)
            {
                if (glm::dot(up_vec, object_points[i]) + plane.w > GroundThreshold)
                {
                    bb.AddPoint(object_points[i]);
                }
            }

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                const VertexAttrib pos_clr_vertex_attribs[] = {
                    {VertexAttrib::Semantic::Position, 0, 3},
                    {VertexAttrib::Semantic::Color, 0, 3},
                };
                constexpr uint32_t PosAttribIndex = 0;
                constexpr uint32_t ColorAttribIndex = 1;
                const VertexDesc pos_clr_vertex_desc(pos_clr_vertex_attribs);

                {
                    Mesh pc_mesh = Mesh(pos_clr_vertex_desc, 0, 0);

                    for (uint32_t i = 0; i < object_points.size(); ++i)
                    {
                        if (glm::dot(up_vec, object_points[i]) + plane.w > GroundThreshold)
                        {
                            const uint32_t vertex_index = pc_mesh.NumVertices();
                            pc_mesh.ResizeVertices(vertex_index + 1);

                            pc_mesh.VertexData<glm::vec3>(vertex_index, PosAttribIndex) = object_points[i];
                            pc_mesh.VertexData<glm::vec3>(vertex_index, ColorAttribIndex) = object_colors[i];
                        }
                    }

                    SaveMesh(pc_mesh, tmp_dir / "ObjectPoints.ply");
                }
                {
                    Mesh pc_mesh = Mesh(pos_clr_vertex_desc, 0, 0);

                    for (uint32_t i = 0; i < plane_points.size(); ++i)
                    {
                        const uint32_t vertex_index = pc_mesh.NumVertices();
                        pc_mesh.ResizeVertices(vertex_index + 1);

                        pc_mesh.VertexData<glm::vec3>(vertex_index, PosAttribIndex) = plane_points[i];
                        pc_mesh.VertexData<glm::vec3>(vertex_index, ColorAttribIndex) = plane_colors[i];
                    }

                    SaveMesh(pc_mesh, tmp_dir / "PlanePoints.ply");
                }
            }
#endif
        }

        std::vector<GpuTexture2D> RotateImages(const StructureFromMotion::Result& sfm_input, const glm::vec3& up_vec)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();
            auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Render);

            const glm::vec4 target_up_vec_ws = glm::vec4(up_vec.x, up_vec.y, up_vec.z, 0);

            std::vector<GpuTexture2D> rotated_images(sfm_input.projections.size());
            for (uint32_t i = 0; i < sfm_input.projections.size(); ++i)
            {
                const auto& projection = sfm_input.projections[i];

                GpuTexture2D rotated_roi_tex;
                {
                    const uint32_t delighted_width = projection.image->Width(0);
                    const uint32_t delighted_height = projection.image->Height(0);
                    const GpuShaderResourceView delighted_srv(gpu_system, *projection.image);

                    GpuConstantBufferOfType<RotateConstantBuffer> rotation_cb(gpu_system, "rotation_cb");

                    const auto& view_mtx = projection.view_mtx;
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

                    rotation_cb.UploadStaging();
                    const GpuConstantBufferView rotation_cbv(gpu_system, rotation_cb);

                    const float cos_theta = glm::dot(new_right_vec, old_right_vec);
                    const float abs_sin_theta = std::sqrt(1 - cos_theta * cos_theta);
                    const uint32_t rotated_size = static_cast<uint32_t>(std::round(size * (std::abs(cos_theta) + abs_sin_theta)));
                    rotated_roi_tex = GpuTexture2D(gpu_system, rotated_size, rotated_size, 1, ColorFmt,
                        GpuResourceFlag::ShaderResource | GpuResourceFlag::RenderTarget, "rotated_roi_tex");
                    GpuRenderTargetView rotated_roi_rtv(gpu_system, rotated_roi_tex);

                    const float clear_clr[] = {0, 0, 0, 1};
                    cmd_list.Clear(rotated_roi_rtv, clear_clr);

                    std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                        {"param_cb", &rotation_cbv},
                    };
                    std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                        {"image_tex", &delighted_srv},
                    };
                    const GpuCommandList::ShaderBinding shader_bindings[] = {
                        {cbvs, {}, {}},
                        {{}, srvs, {}},
                    };

                    GpuRenderTargetView* rtvs[] = {&rotated_roi_rtv};

                    const GpuViewport viewport = {0.0f, 0.0f, static_cast<float>(rotated_size), static_cast<float>(rotated_size)};
                    cmd_list.Render(rotate_pipeline_, {}, {4}, shader_bindings, rtvs, nullptr, std::span(&viewport, 1), {});
                }

                const uint32_t rotated_width = rotated_roi_tex.Width(0);
                const uint32_t rotated_height = rotated_roi_tex.Height(0);

                GpuTexture2D resized_rotated_roi_x_tex(gpu_system, ResizedImageSize, rotated_height, 1, ColorFmt,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "resized_rotated_roi_x_tex");
                auto& resized_rotated_roi_tex = rotated_images[i];
                resized_rotated_roi_tex = GpuTexture2D(gpu_system, ResizedImageSize, ResizedImageSize, 1, ColorFmt,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable,
                    "resized_rotated_roi_tex");
                {
                    constexpr uint32_t BlockDim = 16;

                    const GpuShaderResourceView input_srv(gpu_system, rotated_roi_tex);
                    GpuUnorderedAccessView output_uav(gpu_system, resized_rotated_roi_x_tex);

                    GpuConstantBufferOfType<ResizeConstantBuffer> downsample_x_cb(gpu_system, "downsample_x_cb");
                    downsample_x_cb->src_roi = glm::uvec4(0, 0, rotated_width, rotated_height);
                    downsample_x_cb->dest_size = glm::uvec2(ResizedImageSize, rotated_height);
                    downsample_x_cb->scale = static_cast<float>(rotated_width) / ResizedImageSize;
                    downsample_x_cb->x_dir = true;
                    downsample_x_cb.UploadStaging();
                    const GpuConstantBufferView downsample_x_cbv(gpu_system, downsample_x_cb);

                    std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                        {"param_cb", &downsample_x_cbv},
                    };
                    std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                        {"input_tex", &input_srv},
                    };
                    std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                        {"output_tex", &output_uav},
                    };
                    const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                    cmd_list.Compute(
                        resize_pipeline_, {DivUp(ResizedImageSize, BlockDim), DivUp(rotated_height, BlockDim), 1}, shader_binding);
                }
                {
                    constexpr uint32_t BlockDim = 16;

                    const GpuShaderResourceView input_srv(gpu_system, resized_rotated_roi_x_tex);
                    GpuUnorderedAccessView output_uav(gpu_system, resized_rotated_roi_tex);

                    GpuConstantBufferOfType<ResizeConstantBuffer> downsample_y_cb(gpu_system, "downsample_y_cb");
                    downsample_y_cb->src_roi = glm::uvec4(0, 0, ResizedImageSize, rotated_height);
                    downsample_y_cb->dest_size = glm::uvec2(ResizedImageSize, ResizedImageSize);
                    downsample_y_cb->scale = static_cast<float>(rotated_height) / ResizedImageSize;
                    downsample_y_cb->x_dir = false;
                    downsample_y_cb.UploadStaging();
                    const GpuConstantBufferView downsample_y_cbv(gpu_system, downsample_y_cb);

                    std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                        {"param_cb", &downsample_y_cbv},
                    };
                    std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                        {"input_tex", &input_srv},
                    };
                    std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                        {"output_tex", &output_uav},
                    };
                    const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                    cmd_list.Compute(
                        resize_pipeline_, {DivUp(ResizedImageSize, BlockDim), DivUp(ResizedImageSize, BlockDim), 1}, shader_binding);
                }

#ifdef AIHI_KEEP_INTERMEDIATES
                // It could be used in a copy queue
                resized_rotated_roi_tex.Transition(cmd_list, GpuResourceState::Common);
#endif
            }

            gpu_system.Execute(std::move(cmd_list));

            return rotated_images;
        }

        GpuMesh GenMesh(std::span<const GpuTexture2D> input_images, GpuTexture3D& color_tex, Gaussians& gaussians)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            GpuCommandList cmd_list;
            uint32_t grid_res;
            GpuTexture3D index_vol_tex;
            GpuBuffer density_features_buff;
            GpuBuffer deformation_features_buff;
            GpuBuffer color_features_buff;

            if (!py_init_finished_)
            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                py_init_future_.wait();

                py_init_finished_ = true;
            }

            {
                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();
                auto& tensor_converter = aihi_.TensorConverterInstance();

                cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);

                const uint32_t num_images = static_cast<uint32_t>(input_images.size());
                auto imgs_args = python_system.MakeTupleOfSize(num_images);
                for (uint32_t i = 0; i < num_images; ++i)
                {
                    auto image_data = MakePyObjectPtr(tensor_converter.ConvertPy(cmd_list, input_images[i]));
                    python_system.SetTupleItem(*imgs_args, i, std::move(image_data));
                }
                gpu_system.ExecuteAndReset(cmd_list);

                python_system.CallObject(*mesh_generator_gen_features_method_, std::move(imgs_args));

                const auto py_grid_res = python_system.CallObject(*mesh_generator_resolution_method_);
                grid_res = python_system.Cast<uint32_t>(*py_grid_res);

                const auto py_coords = python_system.CallObject(*mesh_generator_coords_method_);
                GpuBuffer coords_buff;
                tensor_converter.ConvertPy(
                    cmd_list, *py_coords, coords_buff, GpuHeap::Default, GpuResourceFlag::ShaderResource, "MeshGenerator.coords_buff");
                const GpuShaderResourceView coords_srv(gpu_system, coords_buff, GpuFormat::RGB32_Uint);

                index_vol_tex = GpuTexture3D(gpu_system, grid_res, grid_res, grid_res, 1, GpuFormat::R32_Uint,
                    GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "MeshGenerator.index_vol_tex");
                {
                    const uint32_t num_features = coords_buff.Size() / sizeof(glm::uvec3);

                    GpuConstantBufferOfType<ScatterIndexConstantBuffer> scatter_index_cb(gpu_system, "scatter_index_cb");
                    scatter_index_cb->num_features = num_features;
                    scatter_index_cb.UploadStaging();
                    const GpuConstantBufferView scatter_index_cbv(gpu_system, scatter_index_cb);

                    GpuUnorderedAccessView index_vol_uav(gpu_system, index_vol_tex);
                    const uint32_t zeros[] = {0, 0, 0, 0};
                    cmd_list.Clear(index_vol_uav, zeros);

                    constexpr uint32_t BlockDim = 256;

                    std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                        {"param_cb", &scatter_index_cbv},
                    };
                    std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                        {"coords", &coords_srv},
                    };
                    std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                        {"index_volume", &index_vol_uav},
                    };
                    const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};

                    cmd_list.Compute(scatter_index_pipeline_, {DivUp(num_features, BlockDim), 1, 1}, shader_binding);
                }

                const auto py_density_features = python_system.CallObject(*mesh_generator_density_features_method_);
                tensor_converter.ConvertPy(cmd_list, *py_density_features, density_features_buff, GpuHeap::Default,
                    GpuResourceFlag::ShaderResource, "MeshGenerator.density_features_buff");

                const auto py_deformation_features = python_system.CallObject(*mesh_generator_deformation_features_method_);
                tensor_converter.ConvertPy(cmd_list, *py_deformation_features, deformation_features_buff, GpuHeap::Default,
                    GpuResourceFlag::ShaderResource, "MeshGenerator.deformation_features_buff");

                const auto py_color_features = python_system.CallObject(*mesh_generator_color_features_method_);
                tensor_converter.ConvertPy(cmd_list, *py_color_features, color_features_buff, GpuHeap::Default,
                    GpuResourceFlag::ShaderResource, "MeshGenerator.color_features_buff");

                const auto py_num_gaussians = python_system.CallObject(*mesh_generator_gsplat_num_gaussians_method_);
                gaussians.num_gaussians = python_system.Cast<uint32_t>(*py_num_gaussians);

                const auto py_sh_coefficients = python_system.CallObject(*mesh_generator_gsplat_sh_coefficients_method_);
                gaussians.sh_degrees = static_cast<uint32_t>(std::round(std::sqrt(python_system.Cast<uint32_t>(*py_sh_coefficients)))) - 1;

                const auto py_gsplat_positions = python_system.CallObject(*mesh_generator_gsplat_positions_method_);
                tensor_converter.ConvertPy(cmd_list, *py_gsplat_positions, gaussians.positions, GpuHeap::Default,
                    GpuResourceFlag::ShaderResource, "MeshGenerator.gsplat_positions_buff");

                const auto py_gsplat_scales = python_system.CallObject(*mesh_generator_gsplat_scales_method_);
                tensor_converter.ConvertPy(cmd_list, *py_gsplat_scales, gaussians.scales, GpuHeap::Default, GpuResourceFlag::ShaderResource,
                    "MeshGenerator.gsplat_scales_buff");

                const auto py_gsplat_rotations = python_system.CallObject(*mesh_generator_gsplat_rotations_method_);
                tensor_converter.ConvertPy(cmd_list, *py_gsplat_rotations, gaussians.rotations, GpuHeap::Default,
                    GpuResourceFlag::ShaderResource, "MeshGenerator.gsplat_rotations_buff");

                const auto py_gsplat_shs = python_system.CallObject(*mesh_generator_gsplat_shs_method_);
                tensor_converter.ConvertPy(cmd_list, *py_gsplat_shs, gaussians.shs, GpuHeap::Default, GpuResourceFlag::ShaderResource,
                    "MeshGenerator.gsplat_shs_buff");

                const auto py_gsplat_opacities = python_system.CallObject(*mesh_generator_gsplat_opacities_method_);
                tensor_converter.ConvertPy(cmd_list, *py_gsplat_opacities, gaussians.opacities, GpuHeap::Default,
                    GpuResourceFlag::ShaderResource, "MeshGenerator.gsplat_opacities_buff");
            }

            const GpuShaderResourceView density_features_srv(gpu_system, density_features_buff, GpuFormat::R16_Float);
            const GpuShaderResourceView deformation_features_srv(gpu_system, deformation_features_buff, GpuFormat::R16_Float);
            const GpuShaderResourceView color_features_srv(gpu_system, color_features_buff, GpuFormat::R16_Float);

            const uint32_t size = grid_res + 1;
            GpuTexture3D density_deformation_tex(gpu_system, size, size, size, 1, GpuFormat::RGBA16_Float,
                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "MeshGenerator.density_deformation_tex");
            color_tex = GpuTexture3D(gpu_system, size, size, size, 1, GpuFormat::RGBA8_UNorm,
                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess, "color_tex");

            {
                GpuConstantBufferOfType<GatherVolumeConstantBuffer> gather_volume_cb(gpu_system, "gather_volume_cb");
                gather_volume_cb->grid_res = grid_res;
                gather_volume_cb->size = size;
                gather_volume_cb.UploadStaging();

                const GpuConstantBufferView gather_volume_cbv(gpu_system, gather_volume_cb);
                const GpuShaderResourceView index_vol_srv(gpu_system, index_vol_tex);
                GpuUnorderedAccessView density_deformation_uav(gpu_system, density_deformation_tex);
                GpuUnorderedAccessView color_uav(gpu_system, color_tex);

                const float zeros[] = {0, 0, 0, 0};
                cmd_list.Clear(color_uav, zeros);

                constexpr uint32_t BlockDim = 16;

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &gather_volume_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"index_volume", &index_vol_srv},
                    {"density_features", &density_features_srv},
                    {"deformation_features", &deformation_features_srv},
                    {"color_features", &color_features_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"density_deformation_volume", &density_deformation_uav},
                    {"color_volume", &color_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};

                cmd_list.Compute(gather_volume_pipeline_, {DivUp(size, BlockDim), DivUp(size, BlockDim), size}, shader_binding);
            }

            GpuTexture3D dilated_3d_tmp_gpu_tex(gpu_system, color_tex.Width(0), color_tex.Height(0), color_tex.Depth(0), 1,
                color_tex.Format(), GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess,
                "MeshGenerator.dilated_3d_tmp_gpu_tex");

            GpuTexture3D* dilated_gpu_tex = this->DilateTexture(cmd_list, color_tex, dilated_3d_tmp_gpu_tex);
            if (dilated_gpu_tex != &color_tex)
            {
                color_tex = std::move(*dilated_gpu_tex);
            }

            gpu_system.Execute(std::move(cmd_list));

            GpuMesh pos_only_mesh = marching_cubes_.Generate(density_deformation_tex, 0, GridScale);
            pos_only_mesh = invisible_faces_remover_.Process(pos_only_mesh);
            return this->CleanMesh(pos_only_mesh);
        }

        GpuMesh CleanMesh(const GpuMesh& input_gpu_mesh)
        {
            constexpr float Scale = 1e5f;

            auto& gpu_system = aihi_.GpuSystemInstance();
            auto input_mesh = ToMesh(gpu_system, input_gpu_mesh);

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
            return ToGpuMesh(gpu_system, ret_mesh);
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

                Aabb bb;
                for (const uint32_t vi : new_component_vertices)
                {
                    const auto& pos = mesh.VertexData<glm::vec3>(vi, pos_attrib_index);
                    bb.AddPoint(pos);
                }

                const glm::vec3 diagonal = bb.Size();
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

        GpuMesh ApplyVertexColor(const GpuMesh& mesh, const GpuTexture3D& color_vol_tex)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            const auto& vertex_desc = mesh.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib("POSITION", 0);

            const uint32_t num_vertices = mesh.NumVertices();

            GpuConstantBufferOfType<ApplyVertexColorConstantBuffer> apply_vertex_color_cb(gpu_system, "apply_vertex_color_cb");
            apply_vertex_color_cb->inv_scale = 1 / GridScale;
            apply_vertex_color_cb->num_vertices = num_vertices;
            apply_vertex_color_cb->stride = vertex_desc.SlotStrides()[0] / sizeof(float);
            apply_vertex_color_cb->pos_offset = pos_attrib_index * 3;
            apply_vertex_color_cb.UploadStaging();
            const GpuConstantBufferView apply_vertex_color_cbv(gpu_system, apply_vertex_color_cb);

            const GpuShaderResourceView pos_srv(gpu_system, mesh.VertexBuffer(), GpuFormat::R32_Float);
            const GpuShaderResourceView color_vol_srv(gpu_system, color_vol_tex);

            GpuBuffer pos_color_vb(gpu_system, num_vertices * sizeof(glm::vec3) * 2, GpuHeap::Default,
                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::VertexBuffer |
                    GpuResourceFlag::Shareable,
                "pos_color_vb");
            GpuUnorderedAccessView pos_color_uav(gpu_system, pos_color_vb, GpuFormat::R32_Float);

            constexpr uint32_t BlockDim = 256;

            GpuCommandList cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);

            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                {"param_cb", &apply_vertex_color_cbv},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"color_vol_tex", &color_vol_srv},
                {"pos_vertex_buff", &pos_srv},
            };
            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                {"pos_color_vertex_buff", &pos_color_uav},
            };
            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
            cmd_list.Compute(apply_vertex_color_pipeline_, {DivUp(num_vertices, BlockDim), 1, 1}, shader_binding);

            GpuBuffer ib(gpu_system, mesh.IndexBuffer().Size(), GpuHeap::Default,
                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::IndexBuffer |
                    GpuResourceFlag::Shareable,
                "pos_color_ib");
            cmd_list.Copy(ib, mesh.IndexBuffer());

            gpu_system.Execute(std::move(cmd_list));

            GpuVertexLayout pos_color_layout(gpu_system, std::span<const GpuVertexAttrib>({
                                                             {"POSITION", 0, GpuFormat::RGB32_Float},
                                                             {"COLOR", 0, GpuFormat::RGB32_Float},
                                                         }));
            GpuMesh pos_color_mesh(std::move(pos_color_layout), mesh.IndexFormat());

            pos_color_mesh.VertexBuffer() = std::move(pos_color_vb);
            pos_color_mesh.IndexBuffer() = std::move(ib);

            return pos_color_mesh;
        }

        GpuTexture3D* DilateTexture(GpuCommandList& cmd_list, GpuTexture3D& tex, GpuTexture3D& tmp_tex)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            constexpr uint32_t BlockDim = 16;
            const uint32_t max_size = std::max({tex.Width(0), tex.Height(0), tex.Depth(0)});
            const uint32_t dilate_times = LogNextPowerOf2(max_size);

            GpuConstantBufferOfType<DilateConstantBuffer> dilate_cb(gpu_system, "dilate_cb");
            dilate_cb->texture_size = tex.Width(0);
            dilate_cb.UploadStaging();

            const GpuConstantBufferView dilate_cbv(gpu_system, dilate_cb);
            const GpuShaderResourceView tex_srv(gpu_system, tex);
            const GpuShaderResourceView tmp_tex_srv(gpu_system, tmp_tex);
            GpuUnorderedAccessView tex_uav(gpu_system, tex);
            GpuUnorderedAccessView tmp_tex_uav(gpu_system, tmp_tex);

            GpuTexture3D* texs[] = {&tex, &tmp_tex};
            const GpuShaderResourceView* tex_srvs[] = {&tex_srv, &tmp_tex_srv};
            GpuUnorderedAccessView* tex_uavs[] = {&tex_uav, &tmp_tex_uav};
            for (uint32_t i = 0; i < dilate_times; ++i)
            {
                const uint32_t src = i & 1;
                const uint32_t dst = src ? 0 : 1;

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &dilate_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"input_tex", tex_srvs[src]},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"dilated_tex", tex_uavs[dst]},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(dilate_3d_pipeline_,
                    {DivUp(texs[dst]->Width(0), BlockDim), DivUp(texs[dst]->Height(0), BlockDim), texs[dst]->Depth(0)}, shader_binding);
            }

            if (dilate_times & 1)
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
        PyObjectPtr mesh_generator_gsplat_num_gaussians_method_;
        PyObjectPtr mesh_generator_gsplat_sh_coefficients_method_;
        PyObjectPtr mesh_generator_gsplat_positions_method_;
        PyObjectPtr mesh_generator_gsplat_scales_method_;
        PyObjectPtr mesh_generator_gsplat_rotations_method_;
        PyObjectPtr mesh_generator_gsplat_shs_method_;
        PyObjectPtr mesh_generator_gsplat_opacities_method_;
        std::future<void> py_init_future_;
        bool py_init_finished_ = false;

        InvisibleFacesRemover invisible_faces_remover_;
        MarchingCubes marching_cubes_;

        struct ErosionMaskConstantBuffer
        {
            glm::uvec2 texture_size;
            uint32_t erosion;
            uint32_t channel;
        };
        GpuComputePipeline erosion_dilation_mask_pipeline_;

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

        struct DilateConstantBuffer
        {
            uint32_t texture_size;
            uint32_t padding[3];
        };
        GpuComputePipeline dilate_3d_pipeline_;

        struct ApplyVertexColorConstantBuffer
        {
            uint32_t num_vertices;
            float inv_scale;
            uint32_t stride;
            uint32_t pos_offset;
        };
        GpuComputePipeline apply_vertex_color_pipeline_;

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

    MeshGenerator::Result MeshGenerator::Generate(const StructureFromMotion::Result& sfm_input)
    {
        return impl_->Generate(sfm_input);
    }
} // namespace AIHoloImager
