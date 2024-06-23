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
        void Process(const StructureFromMotion::Result& sfm_input, const std::filesystem::path& tmp_dir)
        {
            this->ToOpenMvs(sfm_input, tmp_dir);
        }

    private:
        void ToOpenMvs(const StructureFromMotion::Result& sfm_input, const std::filesystem::path& tmp_dir)
        {
            // Reference from openMVG/src/software/SfM/export/main_openMVG2openMVS.cpp

            const auto mvs_dir = tmp_dir / "Mvs";
            std::filesystem::create_directories(mvs_dir);

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

            const auto out_images_dir = mvs_dir / "Images";
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

                auto& pose = platform.poses.emplace_back();
                image.poseID = static_cast<uint32_t>(platform.poses.size());
                image.ID = static_cast<uint32_t>(i);
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

            const auto out_mvs_file = mvs_dir / "Temp.mvs";
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
        }
    };

    MeshReconstruction::MeshReconstruction() : impl_(std::make_unique<Impl>())
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
