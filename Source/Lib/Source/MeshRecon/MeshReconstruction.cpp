// Copyright (c) 2024 Minmin Gong
//

#include "MeshReconstruction.hpp"

#include <format>
#include <iostream>

#define _USE_EIGEN
#include <InterfaceMVS.h>

namespace AIHoloImager
{
    class MeshReconstruction::Impl
    {
    public:
        explicit Impl(const std::filesystem::path& exe_path) : exe_dir_(exe_path.parent_path())
        {
        }

        void Process(const StructureFromMotion::Result& sfm_input, const std::filesystem::path& tmp_dir)
        {
            working_dir_ = tmp_dir / "Mvs";
            std::filesystem::create_directories(working_dir_);

            std::string mvs_name = this->ToOpenMvs(sfm_input);
            mvs_name = this->PointCloudDensification(mvs_name);
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
                camera.K = intrinsic.k;
                camera.R = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();
                camera.C = Eigen::Vector3d::Zero();
            }

            const auto out_images_dir = working_dir_ / "Images";
            std::filesystem::create_directories(out_images_dir);

            scene.images.reserve(sfm_input.views.size());
            for (size_t i = 0; i < sfm_input.views.size(); ++i)
            {
                const auto& view = sfm_input.views[i];
                const auto image_path = out_images_dir / std::format("{}.jpg", i);

                SaveTexture(view.image, image_path);

                auto& image = scene.images.emplace_back();
                image.name = image_path.string();
                image.platformID = view.intrinsic_id;
                MVS::Interface::Platform& platform = scene.platforms[image.platformID];
                image.cameraID = 0;

                image.poseID = static_cast<uint32_t>(platform.poses.size());
                image.ID = static_cast<uint32_t>(i);

                auto& pose = platform.poses.emplace_back();
                pose.R = view.rotation;
                pose.C = view.center;
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
                vert.X = landmark.point.cast<float>();
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

        std::string PointCloudDensification(const std::string& mvs_name)
        {
            const std::string output_mvs_name = mvs_name + "_Dense";

            const std::string cmd =
                std::format("{} {}.mvs -o {}.mvs --resolution-level 1 --number-views 8 --process-priority 0 --remove-dmaps 1 -w {}",
                    (exe_dir_ / "DensifyPointCloud").string(), mvs_name, output_mvs_name, working_dir_.string());
            const int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                throw ret;
            }

            return output_mvs_name;
        }

    private:
        std::filesystem::path exe_dir_;
        std::filesystem::path working_dir_;
    };

    MeshReconstruction::MeshReconstruction(const std::filesystem::path& exe_path) : impl_(std::make_unique<Impl>(exe_path))
    {
    }

    MeshReconstruction::~MeshReconstruction() = default;

    MeshReconstruction::MeshReconstruction(MeshReconstruction&& other) noexcept = default;
    MeshReconstruction& MeshReconstruction::operator=(MeshReconstruction&& other) noexcept = default;

    void MeshReconstruction::Process(const StructureFromMotion::Result& sfm_input, const std::filesystem::path& tmp_dir)
    {
        return impl_->Process(sfm_input, tmp_dir);
    }
} // namespace AIHoloImager