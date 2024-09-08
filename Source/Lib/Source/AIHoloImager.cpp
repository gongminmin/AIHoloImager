// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/AIHoloImager.hpp"

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

#include "Gpu/GpuSystem.hpp"
#include "MeshGen/MeshGenerator.hpp"
#include "MeshRecon//MeshReconstruction.hpp"
#include "MvRenderer/MultiViewRenderer.hpp"
#include "Python/PythonSystem.hpp"
#include "SfM/StructureFromMotion.hpp"

namespace
{
    std::filesystem::path ExeDir()
    {
        char exe_path[MAX_PATH];
        GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
        return std::filesystem::path(exe_path).parent_path();
    }
} // namespace

namespace AIHoloImager
{
    class AIHoloImager::Impl
    {
    public:
        explicit Impl(const std::filesystem::path& tmp_dir) : exe_dir_(ExeDir()), tmp_dir_(tmp_dir), python_system_(exe_dir_)
        {
        }

        Mesh Generate(const std::filesystem::path& input_path)
        {
            StructureFromMotion::Result sfm_result;
            {
                StructureFromMotion sfm(exe_dir_, gpu_system_, python_system_);
                sfm_result = sfm.Process(input_path, true, tmp_dir_);
            }

            MeshReconstruction::Result mesh_recon_result;
            {
                MeshReconstruction mesh_recon(exe_dir_);
                mesh_recon_result = mesh_recon.Process(sfm_result, true, 2048, tmp_dir_);
            }

            MultiViewRenderer::Result mv_renderer_result;
            {
                MultiViewRenderer mv_renderer(gpu_system_, python_system_, 320, 320);
                mv_renderer_result = mv_renderer.Render(mesh_recon_result.mesh, tmp_dir_);
            }

            Mesh result_mesh;
            {
                MeshGenerator mesh_gen(exe_dir_, gpu_system_, python_system_);
                result_mesh = mesh_gen.Generate(mv_renderer_result.multi_view_images, 2048, mesh_recon_result, tmp_dir_);
            }

            return result_mesh;
        }

    private:
        std::filesystem::path exe_dir_;
        std::filesystem::path tmp_dir_;

        GpuSystem gpu_system_;
        PythonSystem python_system_;
    };

    AIHoloImager::AIHoloImager(const std::filesystem::path& tmp_dir) : impl_(std::make_unique<Impl>(tmp_dir))
    {
    }
    AIHoloImager::AIHoloImager(AIHoloImager&& rhs) noexcept = default;
    AIHoloImager::~AIHoloImager() noexcept = default;

    AIHoloImager& AIHoloImager::operator=(AIHoloImager&& rhs) noexcept = default;

    Mesh AIHoloImager::Generate(const std::filesystem::path& input_path)
    {
        return impl_->Generate(input_path);
    }
} // namespace AIHoloImager
