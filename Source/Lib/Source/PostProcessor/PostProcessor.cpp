// Copyright (c) 2024 Minmin Gong
//

#include "PostProcessor.hpp"

#include <algorithm>
#include <format>

using namespace DirectX;

namespace AIHoloImager
{
    class PostProcessor::Impl
    {
    public:
        explicit Impl(const std::filesystem::path& exe_dir) : exe_dir_(exe_dir)
        {
        }

        Mesh Process(const MeshReconstruction::Result& recon_input, const Mesh& ai_mesh, uint32_t max_texture_size,
            const std::filesystem::path& tmp_dir)
        {
            // TextureMesh can't handle mesh with texture coordinate. Clear it.
            Mesh pos_only_mesh(static_cast<uint32_t>(ai_mesh.Vertices().size()), static_cast<uint32_t>(ai_mesh.Indices().size()));

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < static_cast<uint32_t>(ai_mesh.Vertices().size()); ++i)
            {
                auto& vertex = pos_only_mesh.Vertex(i);
                vertex.pos = ai_mesh.Vertex(i).pos;
                vertex.texcoord = XMFLOAT2(0, 0);
            }

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < static_cast<uint32_t>(ai_mesh.Indices().size()); ++i)
            {
                pos_only_mesh.Index(i) = ai_mesh.Index(i);
            }

            const XMMATRIX transform_mtx = XMLoadFloat4x4(&recon_input.transform);

            std::vector<XMFLOAT3> rh_positions(pos_only_mesh.Vertices().size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(pos_only_mesh.Vertices().size()); ++i)
            {
                XMFLOAT3 pos = pos_only_mesh.Vertex(i).pos;
                std::swap(pos.y, pos.z);

                XMStoreFloat3(&pos, XMVector3TransformCoord(XMLoadFloat3(&pos), transform_mtx));
                pos.z = -pos.z;

                rh_positions[i] = pos;
            }

            DirectX::BoundingOrientedBox ai_obb;
            BoundingOrientedBox::CreateFromPoints(ai_obb, rh_positions.size(), &rh_positions[0], sizeof(rh_positions[0]));

            const float scale_x = ai_obb.Extents.x / recon_input.obb.Extents.x;
            const float scale_y = ai_obb.Extents.y / recon_input.obb.Extents.y;
            const float scale_z = ai_obb.Extents.z / recon_input.obb.Extents.z;
            const float scale = 1 / std::max({scale_x, scale_y, scale_z});

            for (uint32_t i = 0; i < static_cast<uint32_t>(pos_only_mesh.Vertices().size()); ++i)
            {
                auto& vertex = pos_only_mesh.Vertex(i);

                XMFLOAT3 pos = vertex.pos;
                pos.x *= scale;
                pos.y *= scale;
                pos.z *= scale;
                std::swap(pos.y, pos.z);

                XMStoreFloat3(&pos, XMVector3TransformCoord(XMLoadFloat3(&pos), transform_mtx));
                pos.z = -pos.z;

                vertex.pos = pos;
            }

            auto working_dir = tmp_dir / "Mvs";
            std::filesystem::create_directories(working_dir);

            const std::string tmp_mesh_name = "Temp_Ai";
            SaveMesh(pos_only_mesh, working_dir / (tmp_mesh_name + ".glb"));

            const std::string mesh_name = this->MeshTexturing("Temp", tmp_mesh_name, max_texture_size, working_dir);

            Mesh ret = LoadMesh(working_dir / (mesh_name + ".glb"));

            const XMVECTOR center = XMLoadFloat3(&recon_input.obb.Center);
            const XMMATRIX pre_trans = XMMatrixTranslationFromVector(-center);
            const XMMATRIX pre_rotate =
                XMMatrixRotationQuaternion(XMQuaternionInverse(XMLoadFloat4(&recon_input.obb.Orientation))) * XMMatrixRotationZ(XM_PI / 2);
            const XMMATRIX pre_scale = XMMatrixScaling(1, -1, -1);

            const XMMATRIX adjust_mtx = pre_trans * pre_rotate * pre_scale;

            for (uint32_t i = 0; i < static_cast<uint32_t>(ret.Vertices().size()); ++i)
            {
                auto& pos = ret.Vertex(i).pos;
                pos.z = -pos.z;
                XMStoreFloat3(&pos, XMVector3TransformCoord(XMLoadFloat3(&pos), adjust_mtx));
                pos.z = -pos.z;
            }

            return ret;
        }

    private:
        std::string MeshTexturing(
            const std::string& mvs_name, const std::string& mesh_name, uint32_t max_texture_size, const std::filesystem::path& working_dir)
        {
            const std::string output_mesh_name = mesh_name + "_Texture";

            const std::string cmd = std::format(
                "{} {}.mvs -m {}.glb -o {}.glb --export-type glb --decimate 0.5 --ignore-mask-label 0 --max-texture-size {} -w {}",
                (exe_dir_ / "TextureMesh").string(), mvs_name, mesh_name, output_mesh_name, max_texture_size, working_dir.string());
            const int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                throw std::runtime_error(std::format("TextureMesh fails with {}", ret));
            }

            return output_mesh_name;
        }

    private:
        const std::filesystem::path exe_dir_;
    };

    PostProcessor::PostProcessor(const std::filesystem::path& exe_dir) : impl_(std::make_unique<Impl>(exe_dir))
    {
    }

    PostProcessor::~PostProcessor() noexcept = default;

    PostProcessor::PostProcessor(PostProcessor&& other) noexcept = default;
    PostProcessor& PostProcessor::operator=(PostProcessor&& other) noexcept = default;

    Mesh PostProcessor::Process(
        const MeshReconstruction::Result& recon_input, const Mesh& ai_mesh, uint32_t max_texture_size, const std::filesystem::path& tmp_dir)
    {
        return impl_->Process(recon_input, ai_mesh, max_texture_size, tmp_dir);
    }
} // namespace AIHoloImager
