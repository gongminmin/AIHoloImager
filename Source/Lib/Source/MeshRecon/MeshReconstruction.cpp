// Copyright (c) 2024 Minmin Gong
//

#include "MeshReconstruction.hpp"

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

namespace AIHoloImager
{
    class MeshReconstruction::Impl
    {
    public:
        Impl(const std::filesystem::path& exe_dir, GpuSystem& gpu_system)
            : exe_dir_(exe_dir), gpu_system_(gpu_system), texture_recon_(exe_dir_, gpu_system)
        {
        }

        Result Process(const StructureFromMotion::Result& sfm_input, uint32_t texture_size, const std::filesystem::path& tmp_dir)
        {
            working_dir_ = tmp_dir / "Mvs";
            std::filesystem::create_directories(working_dir_);

            std::string mvs_name = this->ToOpenMvs(sfm_input);
            mvs_name = this->PointCloudDensification(mvs_name, sfm_input);

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
            mesh.ResetVertexDesc(VertexDesc(pos_uv_vertex_attribs));

#ifdef AIHI_KEEP_INTERMEDIATES
            SaveMesh(mesh, working_dir_ / "PosUV.glb");
#endif

            const float inv_max_dim = 1 / std::max({obb.extents.x, obb.extents.y, obb.extents.z});
            const glm::mat4x4 model_mtx = RegularizeTransform(-obb.center, glm::inverse(obb.orientation), glm::vec3(inv_max_dim));
            ret.transform = glm::inverse(model_mtx);

            for (uint32_t i = 0; i < mesh.NumVertices(); ++i)
            {
                auto& transformed_pos = mesh.VertexData<glm::vec3>(i, pos_attrib_index);
                const glm::vec4 p = model_mtx * glm::vec4(transformed_pos, 1);
                transformed_pos = glm::vec3(p) / p.w;
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

            const std::string cmd = std::format("{} {}.mvs -o {}.mvs --resolution-level 1 --number-views 8 --process-priority 0 "
                                                "--remove-dmaps 0 --ignore-mask-label 0 -w {}",
                (exe_dir_ / "DensifyPointCloud").string(), mvs_name, output_mvs_name, working_dir_.string());
            const int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                throw std::runtime_error(std::format("DensifyPointCloud fails with {}", ret));
            }

            return output_mvs_name;
        }

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

        TextureReconstruction texture_recon_;
    };

    MeshReconstruction::MeshReconstruction(const std::filesystem::path& exe_dir, GpuSystem& gpu_system)
        : impl_(std::make_unique<Impl>(exe_dir, gpu_system))
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
