// Copyright (c) 2024 Minmin Gong
//

#include "MeshReconstruction.hpp"

#include <bit>
#include <format>
#include <iostream>
#include <numbers>

#include <InterfaceMVS.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Gpu/GpuCommandList.hpp"
#include "MeshSimp/MeshSimplification.hpp"
#include "TextureRecon/TextureReconstruction.hpp"

// #define USES_OPENMVS
#define USES_FILTERED_POINT_CLOUD

namespace AIHoloImager
{
    class MeshReconstruction::Impl
    {
    public:
        Impl(const std::filesystem::path& exe_dir, GpuSystem& gpu_system, PythonSystem& python_system)
            : exe_dir_(exe_dir), gpu_system_(gpu_system), python_system_(python_system)
#ifdef USES_OPENMVS
              ,
              texture_recon_(exe_dir_, gpu_system)
#endif
        {
#ifndef USES_OPENMVS
            image2depth_module_ = python_system_.Import("Image2Depth");
            image2depth_class_ = python_system_.GetAttr(*image2depth_module_, "Image2Depth");
            image2depth_ = python_system_.CallObject(*image2depth_class_);
            image2depth_process_method_ = python_system_.GetAttr(*image2depth_, "Process");
#endif
        }

        ~Impl()
        {
#ifndef USES_OPENMVS
            auto image2depth_destroy_method = python_system_.GetAttr(*image2depth_, "Destroy");
            python_system_.CallObject(*image2depth_destroy_method);
#endif
        }

        Result Process(const StructureFromMotion::Result& sfm_input, uint32_t texture_size, const std::filesystem::path& tmp_dir)
        {
            working_dir_ = tmp_dir / "Mvs";
            std::filesystem::create_directories(working_dir_);

            std::string mvs_name = this->ToOpenMvs(sfm_input);
            mvs_name = this->PointCloudDensification(mvs_name, sfm_input);

#ifdef USES_OPENMVS
            std::string mesh_name = this->RoughMeshReconstruction(mvs_name);
            mesh_name = this->MeshRefinement(mvs_name, mesh_name);

            Result ret;

            auto& mesh = ret.mesh;

            mesh = LoadMesh(working_dir_ / std::format("{}.ply", mesh_name));

            MeshSimplification mesh_simp;
            mesh = mesh_simp.Process(mesh, 0.5f);

            const uint32_t pos_attrib_index = mesh.MeshVertexDesc().FindAttrib(VertexAttrib::Semantic::Position, 0);

            auto& obb = ret.obb;
            obb = Obb::FromPoints(&mesh.VertexData<glm::vec3>(0, pos_attrib_index), mesh.MeshVertexDesc().Stride(), mesh.NumVertices());

            mesh.ComputeNormals();

            mesh = mesh.UnwrapUv(texture_size, 0);

            auto texture_result = texture_recon_.Process(mesh, glm::identity<glm::mat4x4>(), obb, sfm_input, texture_size, tmp_dir);

            auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);
            Texture tex(texture_result.color_tex.Width(0), texture_result.color_tex.Height(0), ElementFormat::RGBA8_UNorm);
            texture_result.color_tex.Readback(gpu_system_, cmd_list, 0, tex.Data());
            gpu_system_.Execute(std::move(cmd_list));

            mesh.AlbedoTexture() = std::move(tex);

            const VertexAttrib pos_uv_vertex_attribs[] = {
                {VertexAttrib::Semantic::Position, 0, 3},
                {VertexAttrib::Semantic::TexCoord, 0, 2},
            };
            constexpr uint32_t PosAttribIndex = 0;
            mesh.ResetVertexDesc(VertexDesc(pos_uv_vertex_attribs));

    #ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, working_dir_ / "PosUV.glb");
    #endif
#else
            (void)texture_size;

            const VertexAttrib pos_clr_vertex_attribs[] = {
                {VertexAttrib::Semantic::Position, 0, 3},
                {VertexAttrib::Semantic::Normal, 0, 3},
                {VertexAttrib::Semantic::Color, 0, 3},
            };
            constexpr uint32_t PosAttribIndex = 0;
            constexpr uint32_t NormalAttribIndex = 1;
            constexpr uint32_t ColorAttribIndex = 2;
            const VertexDesc pos_clr_vertex_desc(pos_clr_vertex_attribs);

            constexpr float Threshold = 0.005f;

            const auto valid_point = [](const Texture& depth_tex, uint32_t depth_x, uint32_t depth_y, const Texture& image_mask_tex,
                                         uint32_t image_x, uint32_t image_y) {
                const uint32_t cropped_width = depth_tex.Width();
                const uint32_t cropped_height = depth_tex.Height();
                const uint32_t image_width = image_mask_tex.Width();

                const uint32_t left_x = depth_x == 0 ? 0 : depth_x - 1;
                const uint32_t right_x = depth_x == cropped_width - 1 ? cropped_width - 1 : depth_x + 1;
                const uint32_t up_y = depth_y == 0 ? 0 : depth_y - 1;
                const uint32_t down_y = depth_y == cropped_height - 1 ? cropped_height - 1 : depth_y + 1;

                const float* depth_data = reinterpret_cast<const float*>(depth_tex.Data());
                const uint8_t* image_mask_data = reinterpret_cast<const uint8_t*>(image_mask_tex.Data());

                return (image_mask_data[(image_y * image_width + image_x) * 4 + 3] >= 128) &&
                       (std::abs(depth_data[depth_y * cropped_width + left_x] - depth_data[depth_y * cropped_width + right_x]) <
                           Threshold) &&
                       (std::abs(depth_data[up_y * cropped_width + depth_x] - depth_data[down_y * cropped_width + depth_x]) < Threshold);
            };

            const auto in_mask = [](const Texture& image_mask_tex, uint32_t image_x, uint32_t image_y) {
                const uint32_t image_width = image_mask_tex.Width();
                const uint8_t* image_mask_data = reinterpret_cast<const uint8_t*>(image_mask_tex.Data());
                return image_mask_data[(image_y * image_width + image_x) * 4 + 3] >= 128;
            };

            auto view_mtxs = std::make_unique<glm::mat4x4[]>(sfm_input.views.size());
            auto proj_mtxs = std::make_unique<glm::mat4x4[]>(sfm_input.views.size());
            auto view_proj_mtxs = std::make_unique<glm::mat4x4[]>(sfm_input.views.size());
            auto vp_offsets = std::make_unique<glm::vec2[]>(sfm_input.views.size());
            for (uint32_t i = 0; i < sfm_input.views.size(); ++i)
            {
                const auto& view = sfm_input.views[i];
                const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                view_mtxs[i] = CalcViewMatrix(view);
                proj_mtxs[i] = CalcProjMatrix(intrinsic, 0.1f, 30.0f);
                view_proj_mtxs[i] = proj_mtxs[i] * view_mtxs[i];
                vp_offsets[i] = CalcViewportOffset(intrinsic);
            }

            Mesh point_cloud = LoadMesh(working_dir_ / (mvs_name + ".ply"));

            struct Point
            {
                glm::vec3 position = glm::vec3(0, 0, 0);
                glm::vec3 normal = glm::vec3(0, 0, 0);
                glm::vec3 color = glm::vec3(0, 0, 0);
            };

            std::vector<std::vector<Point>> points(sfm_input.views.size());
            std::vector<Texture> merged_depth_texs(sfm_input.views.size());
            std::vector<glm::uvec2> merged_depth_offsets(sfm_input.views.size());
            for (uint32_t i = 0; i < sfm_input.views.size(); ++i)
            {
                const auto& view = sfm_input.views[i];
                const auto& intrinsic = sfm_input.intrinsics[view.intrinsic_id];

                const glm::mat4x4& view_mtx = view_mtxs[i];
                const glm::mat4x4& proj_mtx = proj_mtxs[i];
                const glm::vec2& vp_offset = vp_offsets[i];

                const uint32_t image_width = view.image_mask.Width();
                const uint32_t image_height = view.image_mask.Height();

                Texture densified_depth_tex;
                Texture densified_confidence_tex;
    #ifdef USES_FILTERED_POINT_CLOUD
                this->ProjectPointCloud(point_cloud, view_mtx, view_proj_mtxs[i], image_width, image_height, vp_offset, densified_depth_tex,
                    densified_confidence_tex);
    #else
                this->LoadDMap(i, densified_depth_tex, densified_confidence_tex);
    #endif
    #ifdef AIHI_KEEP_INTERMEDIATES
                SaveTexture(densified_depth_tex, working_dir_ / std::format("Depth_{}.pfm", i));
                SaveTexture(densified_confidence_tex, working_dir_ / std::format("Confidence_{}.png", i));
    #endif

                const uint32_t depth_width = densified_depth_tex.Width();
                const uint32_t depth_height = densified_depth_tex.Height();

                const glm::mat4x4 inv_view_mtx = glm::inverse(view_mtx);
                const glm::mat4x4 inv_proj_mtx = glm::inverse(proj_mtx);

                const double fx = intrinsic.k[0].x;
                const float fov_x = glm::degrees(static_cast<float>(2 * std::atan(image_width / (2 * fx))));

                constexpr uint32_t Gap = 32;
                glm::uvec4 expanded_roi;
                expanded_roi.x = std::max(static_cast<uint32_t>(std::floor(view.roi.x)) - Gap, 0U);
                expanded_roi.y = std::max(static_cast<uint32_t>(std::floor(view.roi.y)) - Gap, 0U);
                expanded_roi.z = std::min(static_cast<uint32_t>(std::ceil(view.roi.z)) + Gap, image_width);
                expanded_roi.w = std::min(static_cast<uint32_t>(std::ceil(view.roi.w)) + Gap, image_height);

                auto args = python_system_.MakeTuple(10);
                {
                    auto image = python_system_.MakeObject(
                        std::span<const std::byte>(reinterpret_cast<const std::byte*>(view.image_mask.Data()), view.image_mask.DataSize()));
                    python_system_.SetTupleItem(*args, 0, std::move(image));

                    python_system_.SetTupleItem(*args, 1, python_system_.MakeObject(image_width));
                    python_system_.SetTupleItem(*args, 2, python_system_.MakeObject(image_height));
                    python_system_.SetTupleItem(*args, 3, python_system_.MakeObject(FormatChannels(view.image_mask.Format())));

                    python_system_.SetTupleItem(*args, 4, python_system_.MakeObject(fov_x));

                    auto sfm_depth = python_system_.MakeObject(std::span<const std::byte>(
                        reinterpret_cast<const std::byte*>(densified_depth_tex.Data()), densified_depth_tex.DataSize()));
                    python_system_.SetTupleItem(*args, 5, std::move(sfm_depth));

                    auto sfm_confidence = python_system_.MakeObject(std::span<const std::byte>(
                        reinterpret_cast<const std::byte*>(densified_confidence_tex.Data()), densified_confidence_tex.DataSize()));
                    python_system_.SetTupleItem(*args, 6, std::move(sfm_confidence));

                    python_system_.SetTupleItem(*args, 7, python_system_.MakeObject(depth_width));
                    python_system_.SetTupleItem(*args, 8, python_system_.MakeObject(depth_height));

                    auto roi = python_system_.MakeTuple(4);
                    python_system_.SetTupleItem(*roi, 0, python_system_.MakeObject(expanded_roi.x));
                    python_system_.SetTupleItem(*roi, 1, python_system_.MakeObject(expanded_roi.y));
                    python_system_.SetTupleItem(*roi, 2, python_system_.MakeObject(expanded_roi.z));
                    python_system_.SetTupleItem(*roi, 3, python_system_.MakeObject(expanded_roi.w));
                    python_system_.SetTupleItem(*args, 9, std::move(roi));
                }

                const auto py_depth_data = python_system_.CallObject(*image2depth_process_method_, *args);

                const uint32_t cropped_width = expanded_roi.z - expanded_roi.x;
                const uint32_t cropped_height = expanded_roi.w - expanded_roi.y;

                merged_depth_offsets[i] = glm::uvec2(expanded_roi.x, expanded_roi.y);

                auto& depth_tex = merged_depth_texs[i];
                depth_tex = Texture(cropped_width, cropped_height, ElementFormat::R32_Float);
                float* depth_data = reinterpret_cast<float*>(depth_tex.Data());
                const auto depth_span = python_system_.ToSpan<const float>(*py_depth_data);
                std::memcpy(depth_data, depth_span.data(), depth_tex.DataSize());
    #ifdef AIHI_KEEP_INTERMEDIATES
                SaveTexture(depth_tex, working_dir_ / std::format("MergedDepth_{}.pfm", i));
    #endif

                points[i].resize(cropped_width * cropped_height);

                std::vector<glm::vec3> pos_ess(cropped_width * cropped_height, glm::vec3(0, 0, 0));
                std::vector<bool> valid_mark(cropped_width * cropped_height);
                for (uint32_t y = 0; y < cropped_height; ++y)
                {
                    for (uint32_t x = 0; x < cropped_width; ++x)
                    {
                        const uint32_t ori_x = x + expanded_roi.x;
                        const uint32_t ori_y = y + expanded_roi.y;

                        valid_mark[y * cropped_width + x] = valid_point(
                            depth_tex, x, y, view.delighted_image, ori_x - view.delighted_offset.x, ori_y - view.delighted_offset.y);

                        glm::vec4 point_ps;
                        point_ps.x = 2 * (ori_x - vp_offset.x) / image_width - 1;
                        point_ps.y = 1 - 2 * (ori_y - vp_offset.y) / image_height;
                        point_ps.z = 1;
                        point_ps.w = 1;

                        const glm::vec3 view_dir = glm::vec3(inv_proj_mtx * point_ps);
                        pos_ess[y * cropped_width + x] = -view_dir * (depth_span[y * cropped_width + x] / view_dir.z);
                    }
                }

                std::vector<glm::vec3> normals(cropped_width * cropped_height, glm::vec3(0, 0, 0));
                for (uint32_t y = 1; y < cropped_height - 1; ++y)
                {
                    for (uint32_t x = 1; x < cropped_width - 1; ++x)
                    {
                        if (valid_mark[y * cropped_width + x])
                        {
                            const glm::vec3& up = pos_ess[(y - 1) * cropped_width + x];
                            const glm::vec3& left = pos_ess[y * cropped_width + (x - 1)];
                            const glm::vec3& down = pos_ess[(y + 1) * cropped_width + x];
                            const glm::vec3& right = pos_ess[y * cropped_width + (x + 1)];

                            const glm::vec3 x_dir = right - left;
                            const glm::vec3 y_dir = up - down;
                            glm::vec3& normal = normals[y * cropped_width + x];
                            normal = glm::normalize(glm::cross(x_dir, y_dir));
                        }
                    }
                }

    #ifdef AIHI_KEEP_INTERMEDIATES
                Texture normal_tex(cropped_width, cropped_height, ElementFormat::RGB8_UNorm);
                uint8_t* normal_data = reinterpret_cast<uint8_t*>(normal_tex.Data());
                for (uint32_t y = 0; y < cropped_height; ++y)
                {
                    for (uint32_t x = 0; x < cropped_width; ++x)
                    {
                        glm::vec3 normal = normals[y * cropped_width + x];
                        normal = normal * 0.5f + 0.5f;

                        normal_data[(y * cropped_width + x) * 3 + 0] =
                            static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(normal.x * 255)), 0, 255));
                        normal_data[(y * cropped_width + x) * 3 + 1] =
                            static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(normal.y * 255)), 0, 255));
                        normal_data[(y * cropped_width + x) * 3 + 2] =
                            static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(normal.z * 255)), 0, 255));
                    }
                }
                SaveTexture(normal_tex, working_dir_ / std::format("MergedNormal_{}.png", i));
    #endif

    #ifdef AIHI_KEEP_INTERMEDIATES
                Mesh view_mesh = Mesh(pos_clr_vertex_desc, 0, 0);
    #endif

                const uint8_t* delighted_image_data = reinterpret_cast<const uint8_t*>(view.delighted_image.Data());
                for (uint32_t y = 0; y < cropped_height; ++y)
                {
                    for (uint32_t x = 0; x < cropped_width; ++x)
                    {
                        const uint32_t ori_x = x + expanded_roi.x;
                        const uint32_t ori_y = y + expanded_roi.y;

                        if (valid_mark[y * cropped_width + x])
                        {
                            const glm::vec3& pos_es = pos_ess[y * cropped_width + x];
                            const glm::vec4 pos_ws = inv_view_mtx * glm::vec4(pos_es, 1);

                            const glm::vec3 pos = glm::vec3(pos_ws) / pos_ws.w;

                            const uint32_t image_offset =
                                (ori_y - view.delighted_offset.y) * view.delighted_image.Width() + ori_x - view.delighted_offset.x;
                            const glm::vec3 color = glm::vec3(static_cast<uint8_t>(delighted_image_data[image_offset * 4 + 0]) / 255.0f,
                                static_cast<uint8_t>(delighted_image_data[image_offset * 4 + 1]) / 255.0f,
                                static_cast<uint8_t>(delighted_image_data[image_offset * 4 + 2]) / 255.0f);

                            const glm::vec3& normal_es = normals[y * cropped_width + x];
                            const glm::vec3 normal = glm::normalize(glm::vec3(inv_view_mtx * glm::vec4(normal_es, 0)));

    #ifdef AIHI_KEEP_INTERMEDIATES
                            const uint32_t vertex_index = view_mesh.NumVertices();
                            view_mesh.ResizeVertices(vertex_index + 1);

                            view_mesh.VertexData<glm::vec3>(vertex_index, PosAttribIndex) = pos;
                            view_mesh.VertexData<glm::vec3>(vertex_index, NormalAttribIndex) = normal;
                            view_mesh.VertexData<glm::vec3>(vertex_index, ColorAttribIndex) = color;
    #endif

                            auto& point = points[i][y * cropped_width + x];
                            point.position = pos;
                            point.normal = normal;
                            point.color = color;
                        }
                    }
                }

    #ifdef AIHI_KEEP_INTERMEDIATES
                SaveMesh(view_mesh, working_dir_ / std::format("PointCloud_{}.ply", i));
    #endif
            }

            constexpr float ConfirmThreshold = 0.005f;

            std::vector<Point> fused_points;
            for (uint32_t j = 0; j < sfm_input.views.size(); ++j)
            {
                for (const auto& pt : points[j])
                {
                    if (glm::dot(pt.normal, pt.normal) > 0.5f)
                    {
                        uint32_t num_confirmed_views = 1;
                        Point fused_point = pt;
                        for (uint32_t i = 0; i < sfm_input.views.size(); ++i)
                        {
                            if (i != j)
                            {
                                const auto& other_view = sfm_input.views[i];

                                const glm::mat4x4& other_view_proj_mtx = view_proj_mtxs[i];
                                const glm::vec2& other_vp_offset = vp_offsets[i];

                                const Texture& other_depth_tex = merged_depth_texs[i];
                                const uint32_t other_cropped_width = other_depth_tex.Width();
                                const uint32_t other_cropped_height = other_depth_tex.Height();
                                const uint32_t other_width = other_view.image_mask.Width();
                                const uint32_t other_height = other_view.image_mask.Height();
                                const float* other_depth_data = reinterpret_cast<const float*>(other_depth_tex.Data());

                                const glm::vec4 other_pos_ps = other_view_proj_mtx * glm::vec4(pt.position, 1);
                                const int32_t other_image_x = static_cast<int32_t>(
                                    std::round((other_pos_ps.x / other_pos_ps.w + 1) * other_width / 2 + other_vp_offset.x));
                                const int32_t other_image_y = static_cast<int32_t>(
                                    std::round((1 - other_pos_ps.y / other_pos_ps.w) * other_height / 2 + other_vp_offset.y));
                                const int32_t other_depth_x = other_image_x - merged_depth_offsets[i].x;
                                const int32_t other_depth_y = other_image_y - merged_depth_offsets[i].y;
                                if ((other_depth_x >= 0) && (other_depth_x < static_cast<int32_t>(other_cropped_width)) &&
                                    (other_depth_y >= 0) && (other_depth_y < static_cast<int32_t>(other_cropped_height)) &&
                                    valid_point(other_depth_tex, other_depth_x, other_depth_y, other_view.delighted_image,
                                        other_image_x - other_view.delighted_offset.x, other_image_y - other_view.delighted_offset.y))
                                {
                                    if (std::abs(other_pos_ps.w - other_depth_data[other_depth_y * other_cropped_width + other_depth_x]) <
                                        ConfirmThreshold)
                                    {
                                        auto& other_point = points[i][other_depth_y * other_cropped_width + other_depth_x];
                                        if (glm::dot(other_point.normal, other_point.normal) > 0.5f)
                                        {
                                            fused_point.position += other_point.position;
                                            fused_point.normal += other_point.normal;
                                            fused_point.color += other_point.color;

                                            other_point.normal = glm::vec3(0, 0, 0);

                                            ++num_confirmed_views;
                                        }
                                    }
                                }
                            }
                        }

                        if (num_confirmed_views >= ConfirmViews)
                        {
                            fused_point.position /= static_cast<float>(num_confirmed_views);
                            fused_point.color /= static_cast<float>(num_confirmed_views);
                            fused_points.emplace_back(std::move(fused_point));
                        }
                    }
                }
            }

            Result ret;
            auto& mesh = ret.mesh;
            mesh = Mesh(pos_clr_vertex_desc, static_cast<uint32_t>(fused_points.size()), 0);

            for (uint32_t i = 0; i < fused_points.size(); ++i)
            {
                const auto& point = fused_points[i];
                mesh.VertexData<glm::vec3>(i, PosAttribIndex) = point.position;
                mesh.VertexData<glm::vec3>(i, NormalAttribIndex) = glm::normalize(point.normal);
                mesh.VertexData<glm::vec3>(i, ColorAttribIndex) = point.color;
            }

            points.clear();

            auto& obb = ret.obb;
            obb = Obb::FromPoints(&mesh.VertexData<glm::vec3>(0, PosAttribIndex), pos_clr_vertex_desc.Stride(), mesh.NumVertices());

    #ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, working_dir_ / "PointCloud.ply");
    #endif
#endif

            const float inv_max_dim = 1 / std::max({obb.extents.x, obb.extents.y, obb.extents.z});
            const glm::mat4x4 model_mtx = RegularizeTransform(-obb.center, glm::inverse(obb.orientation), glm::vec3(inv_max_dim));
            ret.transform = glm::inverse(model_mtx);

            for (uint32_t i = 0; i < mesh.NumVertices(); ++i)
            {
                auto& transformed_pos = mesh.VertexData<glm::vec3>(i, PosAttribIndex);
                const glm::vec4 p = model_mtx * glm::vec4(transformed_pos, 1);
                transformed_pos = glm::vec3(p) / p.w;

                auto& transformed_normal = mesh.VertexData<glm::vec3>(i, NormalAttribIndex);
                const glm::vec4 n = model_mtx * glm::vec4(transformed_normal, 0);
                transformed_normal = glm::normalize(glm::vec3(n));
            }

            return ret;
        }

    private:
        std::string ToOpenMvs(const StructureFromMotion::Result& sfm_input)
        {
            // Reference from openMVG/src/software/SfM/export/main_openMVG2openMVS.cpp

            MVS::Interface scene;

            scene.platforms.reserve(sfm_input.intrinsics.size());
            for (const auto& intrinsic : sfm_input.intrinsics)
            {
                auto& platform = scene.platforms.emplace_back();
                auto& camera = platform.cameras.emplace_back();
                camera.width = intrinsic.width;
                camera.height = intrinsic.height;
                std::memcpy(&camera.K, &intrinsic.k, sizeof(intrinsic.k));
                std::memset(&camera.R, 0, sizeof(camera.R));
                camera.R(0, 0) = camera.R(1, 1) = camera.R(2, 2) = 1;
                camera.C = {0, 0, 0};
            }

            const auto out_images_dir = working_dir_ / "Images";
            std::filesystem::create_directories(out_images_dir);

            scene.images.reserve(sfm_input.views.size());
            for (size_t i = 0; i < sfm_input.views.size(); ++i)
            {
                const auto& view = sfm_input.views[i];
                const auto image_path = out_images_dir / std::format("{}.jpg", i);

                auto& image = scene.images.emplace_back();
                image.name = image_path.string();
                image.platformID = view.intrinsic_id;
                MVS::Interface::Platform& platform = scene.platforms[image.platformID];
                image.cameraID = 0;

                const auto mask_path = image_path.parent_path() / (image_path.stem().string() + ".mask.png");
                image.maskName = mask_path.string();

                Texture image_part(view.image_mask.Width(), view.image_mask.Height(), ElementFormat::RGB8_UNorm);
                Texture mask_part(view.image_mask.Width(), view.image_mask.Height(), ElementFormat::R8_UNorm);
                std::memset(image_part.Data(), 0, image_part.DataSize());
                std::memset(mask_part.Data(), 0, mask_part.DataSize());

                const std::byte* src = view.delighted_image.Data();
                std::byte* image_dst = image_part.Data();
                std::byte* mask_dst = mask_part.Data();
                const uint32_t image_width = view.image_mask.Width();
                const uint32_t cropped_width = view.delighted_image.Width();
                const uint32_t cropped_height = view.delighted_image.Height();
                for (uint32_t y = 0; y < cropped_height; ++y)
                {
                    for (uint32_t x = 0; x < cropped_width; ++x)
                    {
                        const uint32_t src_offset = y * cropped_width + x;
                        const uint32_t dst_offset = (view.delighted_offset.y + y) * image_width + (view.delighted_offset.x + x);
                        image_dst[dst_offset * 3 + 0] = src[src_offset * 4 + 0];
                        image_dst[dst_offset * 3 + 1] = src[src_offset * 4 + 1];
                        image_dst[dst_offset * 3 + 2] = src[src_offset * 4 + 2];
                        mask_dst[dst_offset] = src[src_offset * 4 + 3];
                    }
                }

                SaveTexture(image_part, image_path);
                SaveTexture(mask_part, mask_path);

                image.poseID = static_cast<uint32_t>(platform.poses.size());
                image.ID = static_cast<uint32_t>(i);

                auto& pose = platform.poses.emplace_back();
                std::memcpy(&pose.R, &view.rotation, sizeof(view.rotation));
                pose.C = {view.center.x, view.center.y, view.center.z};
            }

            scene.vertices.reserve(sfm_input.structure.size());
            for (const auto& landmark : sfm_input.structure)
            {
                auto& vert = scene.vertices.emplace_back();
                auto& views = vert.views;
                for (const auto& observation : landmark.obs)
                {
                    auto& view = views.emplace_back();
                    view.imageID = observation.view_id;
                    view.confidence = 0;
                }
                if (views.size() < 2)
                {
                    continue;
                }

                std::sort(views.begin(), views.end(), [](const MVS::Interface::Vertex::View& lhs, const MVS::Interface::Vertex::View& rhs) {
                    return lhs.imageID < rhs.imageID;
                });
                vert.X = {static_cast<float>(landmark.point.x), static_cast<float>(landmark.point.y), static_cast<float>(landmark.point.z)};
            }

            const auto out_mvs_file = working_dir_ / "Temp.mvs";
            if (!MVS::ARCHIVE::SerializeSave(scene, out_mvs_file.string()))
            {
                throw std::runtime_error(std::format("Fail to save the file {}", out_mvs_file.string()));
            }

            std::cout << "Scene saved to OpenMVS interface format:\n"
                      << " # platforms: " << scene.platforms.size() << '\n';
            for (size_t i = 0; i < scene.platforms.size(); ++i)
            {
                std::cout << "  platform ( " << i << " ) # cameras: " << scene.platforms[i].cameras.size() << '\n';
            }
            std::cout << "  " << scene.images.size() << " images\n"
                      << "  " << scene.vertices.size() << " landmarks\n";

            return out_mvs_file.stem().string();
        }

        std::string PointCloudDensification(const std::string& mvs_name, const StructureFromMotion::Result& sfm_input)
        {
            // If the dmap files are not removed, the new ones will merged with old ones, causing degradation of the point cloud quality
            for (uint32_t i = 0; i < sfm_input.views.size(); ++i)
            {
                std::filesystem::remove(working_dir_ / std::format("depth{:04}.dmap", i));
            }

            const std::string output_mvs_name = mvs_name + "_Dense";
#ifdef USES_OPENMVS
            const int resolution_level = 1;
#else
    #ifdef USES_FILTERED_POINT_CLOUD
            const int resolution_level = 1;
    #else
            const int resolution_level = 2;
    #endif
#endif

            const std::string cmd = std::format("{} {}.mvs -o {}.mvs --resolution-level {} --number-views 8 --process-priority 0 "
                                                "--remove-dmaps 0 --ignore-mask-label 0 -w {}",
                (exe_dir_ / "DensifyPointCloud").string(), mvs_name, output_mvs_name, resolution_level, working_dir_.string());
            const int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                throw std::runtime_error(std::format("DensifyPointCloud fails with {}", ret));
            }

            return output_mvs_name;
        }

#ifdef USES_FILTERED_POINT_CLOUD
        void ProjectPointCloud(const Mesh& point_cloud, const glm::mat4x4& view_mtx, const glm::mat4x4& view_proj_mtx, uint32_t width,
            uint32_t height, const glm::vec2& vp_offset, Texture& depth_tex, Texture& confidence_tex)
        {
            depth_tex = Texture(width, height, ElementFormat::R32_Float);
            float* depth_data = reinterpret_cast<float*>(depth_tex.Data());
            std::memset(depth_data, 0, depth_tex.DataSize());

            confidence_tex = Texture(width, height, ElementFormat::R8_UNorm);
            uint8_t* confidence_data = reinterpret_cast<uint8_t*>(confidence_tex.Data());
            std::memset(confidence_data, 0, confidence_tex.DataSize());

            const auto& vertex_desc = point_cloud.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);
            const uint32_t normal_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Normal, 0);
            for (uint32_t i = 0; i < point_cloud.NumVertices(); ++i)
            {
                const glm::vec3 normal = point_cloud.VertexData<glm::vec3>(i, normal_attrib_index);
                const glm::vec4 normal_es = view_mtx * glm::vec4(normal, 0);
                if (normal_es.z > 0)
                {
                    const glm::vec3 pos = point_cloud.VertexData<glm::vec3>(i, pos_attrib_index);
                    const glm::vec4 pos_ps = view_proj_mtx * glm::vec4(pos, 1);

                    const int32_t x = static_cast<int32_t>(std::round((pos_ps.x / pos_ps.w + 1) * width / 2 + vp_offset.x));
                    const int32_t y = static_cast<int32_t>(std::round((1 - pos_ps.y / pos_ps.w) * height / 2 + vp_offset.y));
                    if ((x >= 0) && (x < static_cast<int32_t>(width)) && (y >= 0) && (y < static_cast<int32_t>(height)))
                    {
                        float& target = depth_data[y * width + x];
                        if (target < 1e-6f)
                        {
                            target = pos_ps.w;
                            confidence_data[y * width + x] = 255;
                        }
                        else
                        {
                            target = std::min(target, pos_ps.w);
                        }
                    }
                }
            }
        }
#else
        void LoadDMap(uint32_t index, Texture& depth_tex, Texture& confidence_tex)
        {
            struct DMapHeader
            {
                enum class Info : uint8_t
                {
                    None = 0,
                    Depth = 1U << 0,
                    Normal = 1U << 1,
                    Confidence = 1U << 2,
                    Views = 1U << 3,
                };
                static bool EnumHasFlag(Info en, Info flag)
                {
                    return (static_cast<uint32_t>(en) & static_cast<uint32_t>(flag)) != 0;
                }

                uint8_t name[2];                    // file type
                Info type = Info::None;             // content type
                uint8_t padding = 0;                // reserve
                uint32_t image_width, image_height; // image resolution
                uint32_t depth_width, depth_height; // depth-map resolution
                float depth_min, depth_max;         // depth range for this view

                // The content of the dmap:
                // uint16_t file_name_size;
                // char image_file_name[file_name_size];
                // uint32_t num_neighbor_ids;
                // uint32_t neighbor_ids[num_neighbor_ids];
                // dmat3x3 k;
                // dmat3x3 rotation;
                // dvec3 center;
                // float depth_map[depth_height][depth_width];
                // vec3 normal_map[depth_height][depth_width];
                // float confidence_map[depth_height][depth_width];
                // uint32_t view_map[depth_height][depth_width];
            };

            std::ifstream dmap_file(working_dir_ / std::format("depth{:04}.dmap", index), std::ios_base::binary);

            DMapHeader header;
            dmap_file.read(reinterpret_cast<char*>(&header), sizeof(header));
            assert((header.name[0] == 'D') && (header.name[1] == 'R'));
            assert((header.depth_width > 0) && (header.depth_height > 0));
            assert((header.image_width >= header.depth_width) && (header.image_height >= header.depth_height));

            uint16_t file_name_size;
            dmap_file.read(reinterpret_cast<char*>(&file_name_size), sizeof(file_name_size));
            std::string image_file_name(file_name_size, '\0');
            dmap_file.read(image_file_name.data(), file_name_size);

            uint32_t num_neighbor_ids;
            dmap_file.read(reinterpret_cast<char*>(&num_neighbor_ids), sizeof(num_neighbor_ids));
            std::vector<uint32_t> neighbor_ids(num_neighbor_ids);
            dmap_file.read(reinterpret_cast<char*>(neighbor_ids.data()), neighbor_ids.size() * sizeof(neighbor_ids[0]));

            glm::dmat3x3 k;
            dmap_file.read(reinterpret_cast<char*>(&k), sizeof(k));
            glm::dmat3x3 rotation;
            dmap_file.read(reinterpret_cast<char*>(&rotation), sizeof(rotation));
            glm::dvec3 center;
            dmap_file.read(reinterpret_cast<char*>(&center), sizeof(center));

            depth_tex = Texture(header.depth_width, header.depth_height, ElementFormat::R32_Float);
            if (DMapHeader::EnumHasFlag(header.type, DMapHeader::Info::Depth))
            {
                dmap_file.read(reinterpret_cast<char*>(depth_tex.Data()), depth_tex.DataSize());
            }

            if (DMapHeader::EnumHasFlag(header.type, DMapHeader::Info::Normal))
            {
                dmap_file.seekg(header.depth_width * header.depth_height * sizeof(glm::vec3), std::ios_base::cur);
            }

            confidence_tex = Texture(header.depth_width, header.depth_height, ElementFormat::R8_UNorm);
            uint8_t* confidence_data = reinterpret_cast<uint8_t*>(confidence_tex.Data());
            if (DMapHeader::EnumHasFlag(header.type, DMapHeader::Info::Confidence))
            {
                for (uint32_t y = 0; y < header.depth_height; ++y)
                {
                    for (uint32_t x = 0; x < header.depth_width; ++x)
                    {
                        float confidence;
                        dmap_file.read(reinterpret_cast<char*>(&confidence), sizeof(confidence));
                        confidence_data[y * header.depth_width + x] = confidence >= 0.5f ? 255 : 0;
                    }
                }
            }

            if (DMapHeader::EnumHasFlag(header.type, DMapHeader::Info::Views))
            {
                std::vector<uint32_t> view_masks(header.depth_width * header.depth_height);
                dmap_file.read(reinterpret_cast<char*>(view_masks.data()), view_masks.size() * sizeof(uint32_t));
                for (uint32_t y = 0; y < header.depth_height; ++y)
                {
                    for (uint32_t x = 0; x < header.depth_width; ++x)
                    {
                        if ((confidence_data[y * header.depth_width + x] != 0) &&
                            (static_cast<uint32_t>(std::popcount(view_masks[y * header.depth_width + x])) < ConfirmViews))
                        {
                            confidence_data[y * header.depth_width + x] = 0;
                        }
                    }
                }
            }
        }
#endif

        std::string RoughMeshReconstruction(const std::string& mvs_name)
        {
            const std::string output_mesh_name = mvs_name + "_Mesh";

            const std::string cmd = std::format("{} {}.mvs -o {}.ply --process-priority 0 -w {}", (exe_dir_ / "ReconstructMesh").string(),
                mvs_name, output_mesh_name, working_dir_.string());
            const int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                throw std::runtime_error(std::format("ReconstructMesh fails with {}", ret));
            }

            return output_mesh_name;
        }

        std::string MeshRefinement(const std::string& mvs_name, const std::string& mesh_name)
        {
            const std::string output_mesh_name = mesh_name + "_Refine";

            const std::string cmd =
                std::format("{} {}.mvs -m {}.ply -o {}.ply --scales 1 --gradient-step 25.05 --cuda-device -1 --process-priority 0 -w {}",
                    (exe_dir_ / "RefineMesh").string(), mvs_name, mesh_name, output_mesh_name, working_dir_.string());
            const int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                throw std::runtime_error(std::format("RefineMesh fails with {}", ret));
            }

            return output_mesh_name;
        }

    private:
        const std::filesystem::path exe_dir_;
        std::filesystem::path working_dir_;

        GpuSystem& gpu_system_;
        PythonSystem& python_system_;

#ifdef USES_OPENMVS
        TextureReconstruction texture_recon_;
#else
        PyObjectPtr image2depth_module_;
        PyObjectPtr image2depth_class_;
        PyObjectPtr image2depth_;
        PyObjectPtr image2depth_process_method_;

        static constexpr uint32_t ConfirmViews = 3;
#endif
    };

    MeshReconstruction::MeshReconstruction(const std::filesystem::path& exe_dir, GpuSystem& gpu_system, PythonSystem& python_system)
        : impl_(std::make_unique<Impl>(exe_dir, gpu_system, python_system))
    {
    }

    MeshReconstruction::~MeshReconstruction() noexcept = default;

    MeshReconstruction::MeshReconstruction(MeshReconstruction&& other) noexcept = default;
    MeshReconstruction& MeshReconstruction::operator=(MeshReconstruction&& other) noexcept = default;

    MeshReconstruction::Result MeshReconstruction::Process(
        const StructureFromMotion::Result& sfm_input, uint32_t max_texture_size, const std::filesystem::path& tmp_dir)
    {
        return impl_->Process(sfm_input, max_texture_size, tmp_dir);
    }

    glm::mat4x4 RegularizeTransform(const glm::vec3& translate, const glm::quat& rotation, const glm::vec3& scale)
    {
        const glm::mat4x4 pre_trans = glm::translate(glm::identity<glm::mat4x4>(), translate);
        const glm::mat4x4 pre_rotate = glm::rotate(glm::identity<glm::mat4x4>(), std::numbers::pi_v<float>, glm::vec3(1, 0, 0)) *
                                       glm::rotate(glm::identity<glm::mat4x4>(), -std::numbers::pi_v<float> / 2, glm::vec3(0, 0, 1)) *
                                       glm::mat4_cast(rotation);
        const glm::mat4x4 pre_scale = glm::scale(glm::identity<glm::mat4x4>(), scale);

        return pre_scale * pre_rotate * pre_trans;
    }
} // namespace AIHoloImager
