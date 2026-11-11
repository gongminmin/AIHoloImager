// Copyright (c) 2026 Minmin Gong
//

#include "GSplatUtil.hpp"

#include <fstream>

#include "AIHoloImager/Mesh.hpp"

namespace AIHoloImager
{
    glm::vec3 Sh2Rgb(const glm::vec3* sh) noexcept
    {
        constexpr float C0 = 0.28209479177387814f;
        return sh[0] * C0 + 0.5f;
    }

    void SavePointCloud(GpuSystem& gpu_system, const Gaussians& gaussians, const std::filesystem::path& path)
    {
        const VertexAttrib pos_clr_vertex_attribs[] = {
            {VertexAttrib::Semantic::Position, 0, 3},
            {VertexAttrib::Semantic::Color, 0, 3},
        };
        constexpr uint32_t PosAttribIndex = 0;
        constexpr uint32_t ColorAttribIndex = 1;
        const VertexDesc pos_clr_vertex_desc(pos_clr_vertex_attribs);

        Mesh pc_mesh = Mesh(pos_clr_vertex_desc, gaussians.num_gaussians, 0);

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        auto pos_rb_future = cmd_list.ReadBackAsync(gaussians.positions, [&pc_mesh, &gaussians](const void* src_data) {
            const auto* pos_data = reinterpret_cast<const glm::vec3*>(src_data);
            for (uint32_t i = 0; i < gaussians.num_gaussians; ++i)
            {
                pc_mesh.VertexData<glm::vec3>(i, PosAttribIndex) = pos_data[i] * 2.0f;
            }
        });
        auto color_rb_future = cmd_list.ReadBackAsync(gaussians.shs, [&pc_mesh, &gaussians](const void* src_data) {
            const auto* sh_data = reinterpret_cast<const glm::vec3*>(src_data);
            const uint32_t num_coeffs = NumCoefficientsFromShDegrees(gaussians.sh_degrees);
            for (uint32_t i = 0; i < gaussians.num_gaussians; ++i)
            {
                pc_mesh.VertexData<glm::vec3>(i, ColorAttribIndex) =
                    glm::clamp(Sh2Rgb(&sh_data[i * num_coeffs]), glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
            }
        });

        gpu_system.Execute(std::move(cmd_list));

        pos_rb_future.wait();
        color_rb_future.wait();

        SaveMesh(pc_mesh, path);
    }

    Gaussians LoadGaussians(GpuSystem& gpu_system, const std::filesystem::path& path)
    {
        std::ifstream ifs(path, std::ios_base::binary | std::ios_base::in);

        Gaussians ret;

        ifs.read(reinterpret_cast<char*>(&ret.num_gaussians), sizeof(ret.num_gaussians));
        ifs.read(reinterpret_cast<char*>(&ret.sh_degrees), sizeof(ret.sh_degrees));

        const uint32_t num_coeffs = NumCoefficientsFromShDegrees(ret.sh_degrees);

        ret.positions = GpuBuffer(
            gpu_system, ret.num_gaussians * sizeof(glm::vec3), GpuHeap::Default, GpuResourceFlag::ShaderResource, "gaussians.positions");
        ret.scales = GpuBuffer(
            gpu_system, ret.num_gaussians * sizeof(glm::vec3), GpuHeap::Default, GpuResourceFlag::ShaderResource, "gaussians.scales");
        ret.rotations = GpuBuffer(
            gpu_system, ret.num_gaussians * sizeof(glm::vec4), GpuHeap::Default, GpuResourceFlag::ShaderResource, "gaussians.rotations");
        ret.shs = GpuBuffer(gpu_system, ret.num_gaussians * num_coeffs * sizeof(glm::vec3), GpuHeap::Default,
            GpuResourceFlag::ShaderResource, "gaussians.shs");
        ret.opacities = GpuBuffer(
            gpu_system, ret.num_gaussians * sizeof(float), GpuHeap::Default, GpuResourceFlag::ShaderResource, "gaussians.opacities");

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        cmd_list.Upload(ret.positions,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(glm::vec3)); });
        cmd_list.Upload(ret.scales,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(glm::vec3)); });
        cmd_list.Upload(ret.rotations,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(glm::vec4)); });
        cmd_list.Upload(ret.shs, [&ifs, &ret, num_coeffs](void* dst_data) {
            ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * num_coeffs * sizeof(glm::vec3));
        });
        cmd_list.Upload(ret.opacities,
            [&ifs, &ret](void* dst_data) { ifs.read(reinterpret_cast<char*>(dst_data), ret.num_gaussians * sizeof(float)); });

        gpu_system.Execute(std::move(cmd_list));

        return ret;
    }

    void SaveGaussians(GpuSystem& gpu_system, const Gaussians& gaussians, const std::filesystem::path& path)
    {
        std::ofstream ofs(path, std::ios_base::binary | std::ios_base::out);

        ofs.write(reinterpret_cast<const char*>(&gaussians.num_gaussians), sizeof(gaussians.num_gaussians));
        ofs.write(reinterpret_cast<const char*>(&gaussians.sh_degrees), sizeof(gaussians.sh_degrees));

        auto cmd_list = gpu_system.CreateCommandList(GpuSystem::CmdQueueType::Copy);

        auto pos_rb_future = cmd_list.ReadBackAsync(gaussians.positions, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(glm::vec3));
        });
        auto scales_rb_future = cmd_list.ReadBackAsync(gaussians.scales, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(glm::vec3));
        });
        auto rotations_rb_future = cmd_list.ReadBackAsync(gaussians.rotations, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(glm::vec4));
        });
        auto shs_rb_future = cmd_list.ReadBackAsync(gaussians.shs, [&ofs, &gaussians](const void* src_data) {
            const uint32_t num_coeffs = NumCoefficientsFromShDegrees(gaussians.sh_degrees);
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * num_coeffs * sizeof(glm::vec3));
        });
        auto opacities_rb_future = cmd_list.ReadBackAsync(gaussians.opacities, [&ofs, &gaussians](const void* src_data) {
            ofs.write(reinterpret_cast<const char*>(src_data), gaussians.num_gaussians * sizeof(float));
        });

        gpu_system.Execute(std::move(cmd_list));

        pos_rb_future.wait();
        scales_rb_future.wait();
        rotations_rb_future.wait();
        shs_rb_future.wait();
        opacities_rb_future.wait();
    }
} // namespace AIHoloImager
