// Copyright (c) 2026 Minmin Gong
//

#include "MeshOptimizer.hpp"

#include <cassert>
#include <future>
#include <iostream>
#include <map>
#include <tuple>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#ifndef GLM_ENABLE_EXPERIMENTAL
    #define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

#include "Base/Util.hpp"
#include "DiffOptimizer/DiffOptimizer.hpp"
#include "GSplat/GaussianSplatting.hpp"
#include "MeshSimp/MeshSimplification.hpp"
#include "SuperRes/SuperResolution.hpp"
#include "TextureRecon/TextureReconstruction.hpp"
#include "Util/BoundingBox.hpp"
#include "Util/CameraUtil.hpp"
#include "Util/PerfProfiler.hpp"

#include "CompiledShader/MeshOpt/DilateCs.h"
#include "CompiledShader/MeshOpt/ExtractMaskCs.h"
#include "CompiledShader/MeshOpt/MergeTextureCs.h"
#include "CompiledShader/MeshOpt/TransformMeshCs.h"

namespace AIHoloImager
{
    glm::vec2 CalcNearFarPlane(const glm::mat4x4& view_mtx, const Obb& obb)
    {
        glm::vec3 corners[8];
        Obb::GetCorners(obb, corners);

        const glm::vec4 z_col(view_mtx[0].z, view_mtx[1].z, view_mtx[2].z, view_mtx[3].z);

        float min_z_es = std::numeric_limits<float>::max();
        float max_z_es = std::numeric_limits<float>::lowest();
        for (const auto& corner : corners)
        {
            const glm::vec4 pos(corner.x, corner.y, corner.z, 1);
            const float z = glm::dot(pos, z_col);
            min_z_es = std::min(min_z_es, z);
            max_z_es = std::max(max_z_es, z);
        }

        const float center_es_z = (max_z_es + min_z_es) / 2;
        const float extent_es_z = (max_z_es - min_z_es) / 2 * 1.05f;

        const float near_plane = center_es_z + extent_es_z;
        const float far_plane = center_es_z - extent_es_z;
        return glm::vec2(-near_plane, -far_plane);
    }

    class MeshOptimizer::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi)
            : aihi_(aihi), super_res_(aihi), texture_recon_(aihi), gsplat_(aihi), diff_optimizer_(aihi)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            const GpuStaticSampler trilinear_sampler(
                gpu_system, {GpuStaticSampler::Filter::Linear, GpuStaticSampler::Filter::Linear}, GpuStaticSampler::AddressMode::Clamp);
            {
                const ShaderInfo shader = {DEFINE_SHADER(MergeTextureCs)};
                merge_texture_pipeline_ = GpuComputePipeline(gpu_system, shader, std::span{&trilinear_sampler, 1});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(DilateCs)};
                dilate_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(ExtractMaskCs)};
                extract_mask_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }
            {
                const ShaderInfo shader = {DEFINE_SHADER(TransformMeshCs)};
                transform_mesh_pipeline_ = GpuComputePipeline(gpu_system, shader, {});
            }

#ifdef AIHI_KEEP_INTERMEDIATES
            tmp_dir_ = aihi_.TmpDir() / "MeshOpt";
            std::filesystem::create_directories(tmp_dir_);
#endif
        }

        GpuMesh Optimize(const StructureFromMotion::Result& sfm_input, const MeshGenerator::Result& mg_input, uint32_t texture_size)
        {
            auto& profiler = aihi_.PerfProfilerInstance();
            PerfRegion process_perf(profiler, "Mesh optimizer process");

            auto [obb, model_mtx, local_up_vec] = this->OptimizeTransform(sfm_input, mg_input);
            GpuMesh mesh = this->SimplifyMesh(mg_input.mesh);
            this->UvUnwrap(mesh, texture_size);

            const auto gsplat_texture_result = this->GSplatTexture(mesh, mg_input.gaussians, texture_size);
            auto updated_projections = this->SuperResolutionPhotos(sfm_input);
            this->UpdateNearFarPlanes(updated_projections, obb, model_mtx);
            auto photo_texture_result = this->ProjectTexture(mesh, model_mtx, std::span(updated_projections), texture_size);
            const auto mask_gpu_tex = this->ApplyTexture(mesh, gsplat_texture_result, photo_texture_result);
            if (updated_projections.size() > 1)
            {
                this->OptimizeTexture(mesh, model_mtx, std::span(updated_projections), mask_gpu_tex);
            }

            this->AlignMesh(mesh, model_mtx, local_up_vec);

            return mesh;
        }

    private:
        std::tuple<Obb, glm::mat4x4, glm::vec3> OptimizeTransform(
            const StructureFromMotion::Result& sfm_input, const MeshGenerator::Result& mg_input)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();
            auto& profiler = aihi_.PerfProfilerInstance();

            std::cout << "Optimizing transform...\n";

            PerfRegion opt_perf(profiler, "Optimize transform");

            Obb obb;
            {
                const Mesh cpu_mesh = ToMesh(gpu_system, mg_input.mesh);
                const auto& vertex_desc = cpu_mesh.MeshVertexDesc();
                const uint32_t pos_attrib_index = vertex_desc.FindAttrib(VertexAttrib::Semantic::Position, 0);
                obb = Obb::FromPoints(
                    &cpu_mesh.VertexData<glm::vec3>(0, pos_attrib_index), vertex_desc.Stride(), mg_input.mesh.NumVertices());
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
                    if (abs_local_y.y > abs_local_y.z)
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

            glm::mat4x4 model_mtx = this->GuessModelMatrix(obb, mg_input.obj_aabb, local_up_vec, mg_input.up_vec);

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                GpuMesh before_opt_mesh = CopyGpuMesh(gpu_system, mg_input.mesh);
                this->TransformMesh(before_opt_mesh, model_mtx);
                SaveMesh(ToMesh(gpu_system, before_opt_mesh), tmp_dir_ / "BeforeOpt.glb");
            }
            {
                const auto before_opt_obb = Obb::Transform(obb, model_mtx);

                glm::vec3 corners[8];
                Obb::GetCorners(before_opt_obb, corners);

                const Mesh bb_mesh = BoxMesh(corners);
                SaveMesh(bb_mesh, tmp_dir_ / "BeforeOptObb.glb");
            }
#endif

            diff_optimizer_.OptimizeTransform(
                mg_input.mesh, model_mtx, std::span(sfm_input.projections), sfm_input.projections.size() == 1);

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                GpuMesh after_opt_mesh = CopyGpuMesh(gpu_system, mg_input.mesh);
                this->TransformMesh(after_opt_mesh, model_mtx);
                SaveMesh(ToMesh(gpu_system, after_opt_mesh), tmp_dir_ / "AfterOpt.glb");
            }
            {
                const auto after_opt_obb = Obb::Transform(obb, model_mtx);

                glm::vec3 corners[8];
                Obb::GetCorners(after_opt_obb, corners);

                const Mesh bb_mesh = BoxMesh(corners);
                SaveMesh(bb_mesh, tmp_dir_ / "AfterOptObb.glb");
            }
#endif

            return {std::move(obb), std::move(model_mtx), std::move(local_up_vec)};
        }

        glm::mat4x4 GuessModelMatrix(const Obb& obb, const Aabb& obj_aabb, const glm::vec3& local_up_vec, const glm::vec3& up_vec)
        {
            const float diag_len = glm::length(obj_aabb.Size());
            const float scale = diag_len / (glm::length(obb.extents) * 2);

            return glm::translate(glm::identity<glm::mat4x4>(), obj_aabb.Center()) *
                   glm::mat4_cast(glm::normalize(glm::rotation(local_up_vec, up_vec))) *
                   glm::scale(glm::identity<glm::mat4x4>(), glm::vec3(scale));
        }

        GpuMesh SimplifyMesh(const GpuMesh& mesh)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();
            auto& profiler = aihi_.PerfProfilerInstance();

            std::cout << "Simplifying mesh...\n";

            PerfRegion simplify_perf(profiler, "Simplify mesh");

            Mesh cpu_mesh = ToMesh(gpu_system, mesh);

            MeshSimplification mesh_simp;
            cpu_mesh = mesh_simp.Process(cpu_mesh, 0.125f);
            this->FillHoles(cpu_mesh);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(cpu_mesh, tmp_dir_ / "AiMeshSimplified.glb");
#endif

            cpu_mesh.ComputeNormals();

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(cpu_mesh, tmp_dir_ / "AiMeshPosNormal.glb");
#endif

            return ToGpuMesh(gpu_system, cpu_mesh);
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

        void UvUnwrap(GpuMesh& mesh, uint32_t texture_size)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();
            auto& profiler = aihi_.PerfProfilerInstance();

            std::cout << "Unwrapping UV...\n";

            PerfRegion unwrap_perf(profiler, "Unwrap UV");

            Mesh cpu_mesh = ToMesh(gpu_system, mesh);
            cpu_mesh = cpu_mesh.UnwrapUv(texture_size, 2);
            mesh = ToGpuMesh(gpu_system, cpu_mesh);
        }

        TextureReconstruction::Result GSplatTexture(const GpuMesh& mesh, const Gaussians& gaussians, uint32_t texture_size)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();
            auto& profiler = aihi_.PerfProfilerInstance();

            std::cout << "Generating texture from Gaussian splatting...\n";

            PerfRegion gsplat_texturing_perf(profiler, "Generate GSplat texture");

            constexpr uint32_t GSplatNumViews = 8;
            constexpr uint32_t GSplatRenderedSize = 1024;
            auto gsplat_projections = std::make_unique<AIHoloImagerInternal::ProjectionDesc[]>(GSplatNumViews);
            for (uint32_t i = 0; i < GSplatNumViews; ++i)
            {
                auto& projection = gsplat_projections[i];

                projection.full_width = GSplatRenderedSize;
                projection.full_height = GSplatRenderedSize;
                projection.vp_offset = glm::vec2(0, 0);
                projection.image_offset = glm::uvec2(0, 0);

                projection.image =
                    std::make_shared<GpuTexture2D>(gpu_system, GSplatRenderedSize, GSplatRenderedSize, 1, GpuFormat::RGBA8_UNorm_SRGB,
                        GpuResourceFlag::ShaderResource | GpuResourceFlag::RenderTarget | GpuResourceFlag::UnorderedAccess,
                        std::format("gsplat_rendered_{}", i));

                const glm::vec2 angle = SphereHammersleySequence(i, GSplatNumViews);
                const glm::vec3 camera_pos = SphericalCameraPose(angle.x, angle.y, 1.5f);
                const glm::vec3 camera_dir = -glm::normalize(camera_pos);
                glm::vec3 camera_up_vec;
                if (std::abs(camera_dir.z) > 0.95f)
                {
                    camera_up_vec = glm::vec3(1, 0, 0);
                }
                else
                {
                    camera_up_vec = glm::vec3(0, 0, 1);
                }

                projection.view_mtx = glm::lookAtRH(camera_pos, glm::vec3(0, 0, 0), camera_up_vec);
                projection.proj_mtx = glm::perspectiveRH_ZO(glm::radians(45.0f), 1.0f, 0.5f, 2.5f);

                {
                    auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Render);

                    GpuRenderTargetView rendered_tex_rtv(gpu_system, *projection.image);

                    const float bg_clr[] = {0.0f, 0.0f, 0.0f, 1.0f};
                    cmd_list.Clear(rendered_tex_rtv, bg_clr);

                    gpu_system.Execute(std::move(cmd_list));
                }

                gsplat_.Render(gaussians, projection.view_mtx, projection.proj_mtx, 0.1f, *projection.image);

#ifdef AIHI_KEEP_INTERMEDIATES
                {
                    auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

                    const uint32_t width = projection.image->Width(0);
                    const uint32_t height = projection.image->Height(0);
                    Texture gsplat_rb_tex(width, height, ElementFormat::RGBA8_UNorm_SRGB);
                    auto rb_future = cmd_list.ReadBackAsync(*projection.image, 0, gsplat_rb_tex.Data(), gsplat_rb_tex.DataSize());
                    gpu_system.Execute(std::move(cmd_list));

                    rb_future.wait();

                    SaveTexture(gsplat_rb_tex, tmp_dir_ / std::format("RenderedGSplat_{}.png", i));
                }
#endif
            }

            TextureReconstruction::Result gsplat_texture_result =
                texture_recon_.Process(mesh, glm::scale(glm::identity<glm::mat4x4>(), glm::vec3(0.5f)),
                    std::span(gsplat_projections.get(), GSplatNumViews), texture_size);

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

                GpuTexture2D dump_tex(
                    gpu_system, texture_size, texture_size, 1, GpuFormat::RGBA8_UNorm_SRGB, GpuResourceFlag::None, "dump_tex");
                cmd_list.Copy(dump_tex, gsplat_texture_result.color_tex);

                gpu_system.Execute(std::move(cmd_list));

                GpuMesh dump_mesh = CopyGpuMesh(gpu_system, mesh);
                dump_mesh.AlbedoTexture() = std::move(dump_tex);

                SaveMesh(ToMesh(gpu_system, dump_mesh), tmp_dir_ / "AiMeshGSplat.glb");
            }
#endif

            return gsplat_texture_result;
        }

        std::vector<AIHoloImagerInternal::ProjectionDesc> SuperResolutionPhotos(const StructureFromMotion::Result& sfm_input)
        {
            constexpr uint32_t MinImageDim = 1024;

#ifdef AIHI_KEEP_INTERMEDIATES
            auto& gpu_system = aihi_.GpuSystemInstance();
#endif
            auto& profiler = aihi_.PerfProfilerInstance();

            std::cout << "Super resolution...\n";

            std::vector<AIHoloImagerInternal::ProjectionDesc> updated_projections(sfm_input.projections.size());

            PerfRegion super_res_perf(profiler, "Super resolution");

            for (size_t i = 0; i < sfm_input.projections.size(); ++i)
            {
                std::cout << std::format("Upsampling images ({} / {})\r", i + 1, sfm_input.projections.size());

                if ((sfm_input.projections[i].image->Width(0) < MinImageDim) || (sfm_input.projections[i].image->Height(0) < MinImageDim))
                {
                    updated_projections[i] = super_res_.Process(sfm_input.projections[i], 2);
                }
                else
                {
                    updated_projections[i] = sfm_input.projections[i];
                }

#ifdef AIHI_KEEP_INTERMEDIATES
                {
                    auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

                    const uint32_t width = updated_projections[i].image->Width(0);
                    const uint32_t height = updated_projections[i].image->Height(0);
                    Texture super_res_rb_tex(width, height, ElementFormat::RGBA8_UNorm_SRGB);
                    auto rb_future =
                        cmd_list.ReadBackAsync(*updated_projections[i].image, 0, super_res_rb_tex.Data(), super_res_rb_tex.DataSize());
                    gpu_system.Execute(std::move(cmd_list));

                    rb_future.wait();

                    SaveTexture(super_res_rb_tex, tmp_dir_ / std::format("SuperRes_{}.png", i));
                }
#endif
            }
            std::cout << '\n';

            return updated_projections;
        }

        void UpdateNearFarPlanes(
            std::vector<AIHoloImagerInternal::ProjectionDesc>& projections, const Obb& obb, const glm::mat4x4& model_mtx)
        {
            const Obb world_obb = Obb::Transform(obb, model_mtx);
            for (size_t i = 0; i < projections.size(); ++i)
            {
                auto& projection = projections[i];

                const glm::vec2 near_far_plane = CalcNearFarPlane(projection.view_mtx, world_obb);
                const float range = near_far_plane.y - near_far_plane.x;
                projection.proj_mtx[2][2] = -near_far_plane.y / range;
                projection.proj_mtx[3][2] = -(near_far_plane.y * near_far_plane.x) / range;
            }
        }

        TextureReconstruction::Result ProjectTexture(const GpuMesh& mesh, const glm::mat4x4& model_mtx,
            std::span<const AIHoloImagerInternal::ProjectionDesc> projections, uint32_t texture_size)
        {
            auto& profiler = aihi_.PerfProfilerInstance();

            std::cout << "Generating texture from photos...\n";

            PerfRegion photo_texturing_perf(profiler, "Generate photo texture");

            return texture_recon_.Process(mesh, model_mtx, projections, texture_size);
        }

        GpuTexture2D ApplyTexture(
            GpuMesh& mesh, const TextureReconstruction::Result& gsplat_texture_result, TextureReconstruction::Result& photo_texture_result)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();
            auto& profiler = aihi_.PerfProfilerInstance();

            std::cout << "Merging textures...\n";

            PerfRegion merge_textures_perf(profiler, "Merge textures");

            const uint32_t texture_size = photo_texture_result.color_tex.Width(0);

            GpuTexture2D albedo_gpu_tex(gpu_system, texture_size, texture_size, 1, GpuFormat::RGBA8_UNorm_SRGB,
                GpuResourceFlag::ShaderResource | GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, "albedo_gpu_tex");
            GpuTexture2D mask_gpu_tex(gpu_system, texture_size, texture_size, 1, GpuFormat::R8_UNorm,
                GpuResourceFlag::UnorderedAccess | GpuResourceFlag::Shareable, "mask_gpu_tex");

            auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);

            this->MergeTexture(cmd_list, gsplat_texture_result.color_tex, photo_texture_result.color_tex);

            {
                constexpr uint32_t BlockDim = 16;

                const GpuShaderResourceView input_srv(gpu_system, photo_texture_result.color_tex);
                GpuUnorderedAccessView mask_uav(gpu_system, mask_gpu_tex);

                GpuConstantBufferOfType<ExtractMaskConstantBuffer> extract_mask_cb(gpu_system, "extract_mask_cb");
                extract_mask_cb->texture_size = {texture_size, texture_size};
                extract_mask_cb.UploadStaging();
                const GpuConstantBufferView extract_mask_cbv(gpu_system, extract_mask_cb);

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &extract_mask_cbv},
                };
                std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                    {"input_tex", &input_srv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"mask_tex", &mask_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
                cmd_list.Compute(extract_mask_pipeline_, {DivUp(texture_size, BlockDim), DivUp(texture_size, BlockDim), 1}, shader_binding);
            }

            GpuTexture2D* dilated_gpu_tex = this->DilateTexture(cmd_list, photo_texture_result.color_tex, albedo_gpu_tex);
            if (dilated_gpu_tex != &albedo_gpu_tex)
            {
                cmd_list.Copy(albedo_gpu_tex, *dilated_gpu_tex);
            }
            albedo_gpu_tex.Transition(cmd_list, GpuResourceState::Common);

            gpu_system.Execute(std::move(cmd_list));

            mesh.AlbedoTexture() = std::move(albedo_gpu_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(ToMesh(gpu_system, mesh), tmp_dir_ / "AiMeshTextured.glb");
#endif

            return mask_gpu_tex;
        }

        void MergeTexture(GpuCommandList& cmd_list, const GpuTexture2D& gsplat_color_tex, GpuTexture2D& merged_tex)
        {
            assert((gsplat_color_tex.Width(0) == merged_tex.Width(0)) && (gsplat_color_tex.Height(0) == merged_tex.Height(0)));

            auto& gpu_system = aihi_.GpuSystemInstance();

            GpuUnorderedAccessView merged_uav(gpu_system, merged_tex, ToLinearFormat(merged_tex.Format()));

            const uint32_t texture_width = merged_tex.Width(0);
            const uint32_t texture_height = merged_tex.Height(0);

            GpuConstantBufferOfType<MergeTextureConstantBuffer> merge_texture_cb(gpu_system, "merge_texture_cb");
            merge_texture_cb->texture_size = {texture_width, texture_height};
            merge_texture_cb.UploadStaging();

            const GpuConstantBufferView merge_texture_cbv(gpu_system, merge_texture_cb);
            const GpuShaderResourceView gsplat_color_srv(gpu_system, gsplat_color_tex);

            constexpr uint32_t BlockDim = 16;

            std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                {"param_cb", &merge_texture_cbv},
            };
            std::tuple<std::string_view, const GpuShaderResourceView*> srvs[] = {
                {"gsplat_color_tex", &gsplat_color_srv},
            };
            std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                {"merged_tex", &merged_uav},
            };
            const GpuCommandList::ShaderBinding shader_binding = {cbvs, srvs, uavs};
            cmd_list.Compute(merge_texture_pipeline_, {DivUp(texture_width, BlockDim), DivUp(texture_height, BlockDim), 1}, shader_binding);
        }

        GpuTexture2D* DilateTexture(GpuCommandList& cmd_list, GpuTexture2D& tex, GpuTexture2D& tmp_tex)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            constexpr uint32_t BlockDim = 16;
            const uint32_t max_size = std::max(tex.Width(0), tex.Height(0));
            const uint32_t dilate_times = LogNextPowerOf2(max_size);

            GpuConstantBufferOfType<DilateConstantBuffer> dilate_cb(gpu_system, "dilate_cb");
            dilate_cb->texture_size = tex.Width(0);
            dilate_cb.UploadStaging();

            const GpuConstantBufferView dilate_cbv(gpu_system, dilate_cb);
            const GpuShaderResourceView tex_srv(gpu_system, tex);
            const GpuShaderResourceView tmp_tex_srv(gpu_system, tmp_tex);
            GpuUnorderedAccessView tex_uav(gpu_system, tex, ToLinearFormat(tex.Format()));
            GpuUnorderedAccessView tmp_tex_uav(gpu_system, tmp_tex, ToLinearFormat(tmp_tex.Format()));

            GpuTexture2D* texs[] = {&tex, &tmp_tex};
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
                cmd_list.Compute(dilate_pipeline_,
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

        void OptimizeTexture(GpuMesh& mesh, const glm::mat4x4& model_mtx, std::span<const AIHoloImagerInternal::ProjectionDesc> projections,
            const GpuTexture2D& mask_gpu_tex)
        {
            auto& profiler = aihi_.PerfProfilerInstance();

            std::cout << "Optimizing texture...\n";

#ifdef AIHI_KEEP_INTERMEDIATES
            auto& gpu_system = aihi_.GpuSystemInstance();
            const uint32_t texture_size = mesh.AlbedoTexture().Width(0);

            {
                auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

                Texture albedo_tex(texture_size, texture_size, ElementFormat::RGBA8_UNorm_SRGB);
                const auto rb_future = cmd_list.ReadBackAsync(mesh.AlbedoTexture(), 0, albedo_tex.Data(), albedo_tex.DataSize());
                gpu_system.Execute(std::move(cmd_list));
                rb_future.wait();
                SaveTexture(albedo_tex, tmp_dir_ / "BeforeOpt.png");
            }
#endif

            PerfRegion opt_texture_perf(profiler, "Optimize texture");

            diff_optimizer_.OptimizeTexture(mesh, model_mtx, projections, mask_gpu_tex);

#ifdef AIHI_KEEP_INTERMEDIATES
            {
                auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

                Texture albedo_tex(texture_size, texture_size, ElementFormat::RGBA8_UNorm_SRGB);
                const auto rb_future = cmd_list.ReadBackAsync(mesh.AlbedoTexture(), 0, albedo_tex.Data(), albedo_tex.DataSize());
                gpu_system.Execute(std::move(cmd_list));
                rb_future.wait();
                SaveTexture(albedo_tex, tmp_dir_ / "AfterOpt.png");
            }
#endif
        }

        void AlignMesh(GpuMesh& mesh, const glm::mat4x4& model_mtx, const glm::vec3& local_up_vec)
        {
            auto& profiler = aihi_.PerfProfilerInstance();

            PerfRegion aligning_perf(profiler, "Align mesh");

            glm::vec3 scale;
            glm::quat rotation;
            glm::vec3 translation;
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(model_mtx, scale, rotation, translation, skew, perspective);

            const glm::mat4x4 adjust_mtx = glm::recompose(
                scale, glm::normalize(glm::rotation(local_up_vec, glm::vec3(0, 1, 0))), glm::zero<glm::vec3>(), skew, perspective);

            this->TransformMesh(mesh, adjust_mtx);

#ifdef AIHI_KEEP_INTERMEDIATES
            auto& gpu_system = aihi_.GpuSystemInstance();
            SaveMesh(ToMesh(gpu_system, mesh), tmp_dir_ / "AiMesh.glb");
#endif
        }

        void TransformMesh(GpuMesh& mesh, const glm::mat4x4& transform_mtx)
        {
            auto& gpu_system = aihi_.GpuSystemInstance();

            auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Compute);

            const uint32_t num_vertices = mesh.NumVertices();
            const auto& vertex_desc = mesh.MeshVertexDesc();
            const uint32_t pos_attrib_index = vertex_desc.FindAttrib("POSITION", 0);
            const uint32_t normal_attrib_index = vertex_desc.FindAttrib("NORMAL", 0);

            const auto vertex_attribs = vertex_desc.Attribs();
            const uint32_t pos_offset = pos_attrib_index == GpuVertexLayout::InvalidIndex
                                            ? GpuVertexLayout::InvalidIndex
                                            : vertex_attribs[pos_attrib_index].offset / sizeof(float);
            const uint32_t normal_offset = normal_attrib_index == GpuVertexLayout::InvalidIndex
                                               ? GpuVertexLayout::InvalidIndex
                                               : vertex_attribs[normal_attrib_index].offset / sizeof(float);

            {
                constexpr uint32_t BlockDim = 256;

                GpuUnorderedAccessView vertex_buff_uav(gpu_system, mesh.VertexBuffer(), GpuFormat::R32_Float);

                GpuConstantBufferOfType<TransformMeshConstantBuffer> transform_mesh_cb(gpu_system, "transform_mesh_cb");
                transform_mesh_cb->num_vertices = num_vertices;
                transform_mesh_cb->stride = mesh.MeshVertexDesc().SlotStrides()[0] / sizeof(float);
                transform_mesh_cb->pos_offset = pos_offset;
                transform_mesh_cb->normal_offset = normal_offset;
                transform_mesh_cb->transform_mtx = glm::transpose(transform_mtx);
                transform_mesh_cb->transform_it_mtx = glm::inverse(transform_mtx);
                transform_mesh_cb.UploadStaging();
                const GpuConstantBufferView transform_mesh_cbv(gpu_system, transform_mesh_cb);

                std::tuple<std::string_view, const GpuConstantBufferView*> cbvs[] = {
                    {"param_cb", &transform_mesh_cbv},
                };
                std::tuple<std::string_view, GpuUnorderedAccessView*> uavs[] = {
                    {"vertex_buff", &vertex_buff_uav},
                };
                const GpuCommandList::ShaderBinding shader_binding = {cbvs, {}, uavs};
                cmd_list.Compute(transform_mesh_pipeline_, {DivUp(num_vertices, BlockDim), 1, 1}, shader_binding);
            }

            gpu_system.Execute(std::move(cmd_list));
        }

    private:
        AIHoloImagerInternal& aihi_;

#ifdef AIHI_KEEP_INTERMEDIATES
        std::filesystem::path tmp_dir_;
#endif

        DiffOptimizer diff_optimizer_;
        GaussianSplatting gsplat_;
        SuperResolution super_res_;
        TextureReconstruction texture_recon_;

        struct MergeTextureConstantBuffer
        {
            glm::uvec2 texture_size;
            uint32_t padding[2];
        };
        GpuComputePipeline merge_texture_pipeline_;

        struct DilateConstantBuffer
        {
            uint32_t texture_size;
            uint32_t padding[3];
        };
        GpuComputePipeline dilate_pipeline_;

        struct ExtractMaskConstantBuffer
        {
            glm::uvec2 texture_size;
            uint32_t padding[2];
        };
        GpuComputePipeline extract_mask_pipeline_;

        struct TransformMeshConstantBuffer
        {
            uint32_t num_vertices;
            uint32_t stride;
            uint32_t pos_offset;
            uint32_t normal_offset;
            glm::mat4x4 transform_mtx;
            glm::mat4x4 transform_it_mtx;
        };
        GpuComputePipeline transform_mesh_pipeline_;
    };

    MeshOptimizer::MeshOptimizer(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }
    MeshOptimizer::~MeshOptimizer() noexcept = default;

    MeshOptimizer::MeshOptimizer(MeshOptimizer&& other) noexcept = default;
    MeshOptimizer& MeshOptimizer::operator=(MeshOptimizer&& other) noexcept = default;

    GpuMesh MeshOptimizer::Optimize(
        const StructureFromMotion::Result& sfm_input, const MeshGenerator::Result& mg_input, uint32_t texture_size)
    {
        return impl_->Optimize(sfm_input, mg_input, texture_size);
    }
} // namespace AIHoloImager
