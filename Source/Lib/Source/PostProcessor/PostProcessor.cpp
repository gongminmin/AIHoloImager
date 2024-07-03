// Copyright (c) 2024 Minmin Gong
//

#include "PostProcessor.hpp"

#include <algorithm>
#include <format>
#include <set>
#include <tuple>

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
            std::set<std::tuple<int32_t, int32_t, int32_t>> unique_int_positions;
            for (uint32_t i = 0; i < ai_mesh.Vertices().size(); ++i)
            {
                const auto& pos = ai_mesh.Vertex(i).pos;
                std::tuple<int32_t, int32_t, int32_t> int_pos = {static_cast<int32_t>(pos.x * 1e5f + 0.5f),
                    static_cast<uint32_t>(pos.y * 1e5f + 0.5f), static_cast<uint32_t>(pos.z * 1e5f + 0.5f)};
                unique_int_positions.emplace(std::move(int_pos));
            }

            std::vector<std::tuple<int32_t, int32_t, int32_t>> unique_int_positions_vec(
                unique_int_positions.begin(), unique_int_positions.end());
            std::vector<XMFLOAT3> unique_positions_vec(unique_int_positions_vec.size());
            std::vector<uint32_t> unique_position_mapping(ai_mesh.Vertices().size());
#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < ai_mesh.Vertices().size(); ++i)
            {
                const auto& pos = ai_mesh.Vertex(i).pos;
                const std::tuple<int32_t, int32_t, int32_t> int_pos = {static_cast<int32_t>(pos.x * 1e5f + 0.5f),
                    static_cast<uint32_t>(pos.y * 1e5f + 0.5f), static_cast<uint32_t>(pos.z * 1e5f + 0.5f)};

                const uint32_t found_pos_index =
                    static_cast<uint32_t>(std::lower_bound(unique_int_positions_vec.begin(), unique_int_positions_vec.end(), int_pos) -
                                          unique_int_positions_vec.begin());

                unique_positions_vec[found_pos_index] = pos;
                unique_position_mapping[i] = found_pos_index;
            }

            Mesh pos_only_mesh(static_cast<uint32_t>(unique_positions_vec.size()), 0);

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < static_cast<uint32_t>(unique_positions_vec.size()); ++i)
            {
                auto& vertex = pos_only_mesh.Vertex(i);
                vertex.pos.x = unique_positions_vec[i].x;
                vertex.pos.y = unique_positions_vec[i].y;
                vertex.pos.z = unique_positions_vec[i].z;
                vertex.texcoord = XMFLOAT2(0, 0);
            }

            std::vector<uint32_t> indices;
            indices.reserve(ai_mesh.Indices().size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(ai_mesh.Indices().size() / 3); ++i)
            {
                const uint32_t tri[] = {unique_position_mapping[ai_mesh.Index(i * 3 + 0)],
                    unique_position_mapping[ai_mesh.Index(i * 3 + 1)], unique_position_mapping[ai_mesh.Index(i * 3 + 2)]};
                if ((tri[0] != tri[1]) && (tri[1] != tri[2]) && (tri[2] != tri[0]))
                {
                    indices.push_back(tri[0]);
                    indices.push_back(tri[1]);
                    indices.push_back(tri[2]);
                }
            }

            pos_only_mesh.ResizeIndices(static_cast<uint32_t>(indices.size()));
            for (uint32_t i = 0; i < indices.size(); ++i)
            {
                pos_only_mesh.Index(i) = indices[i];
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
