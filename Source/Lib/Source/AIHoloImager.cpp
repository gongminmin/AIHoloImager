// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/AIHoloImager.hpp"

#include <format>
#include <iostream>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h> // For GetModuleFileNameA
#endif

#include "Gpu/GpuSystem.hpp"
#include "MeshGen/MeshGenerator.hpp"
#include "MeshRecon/MeshReconstruction.hpp"
#include "Python/PythonSystem.hpp"
#include "SfM/StructureFromMotion.hpp"
#include "Util/Timer.hpp"

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
            Timer timer;

            std::chrono::duration<double> sfm_init_time;
            std::chrono::duration<double> sfm_time;
            StructureFromMotion::Result sfm_result;
            {
                timer.Restart();
                StructureFromMotion sfm(exe_dir_, gpu_system_, python_system_);
                sfm_init_time = timer.Elapsed();
                sfm_result = sfm.Process(input_path, true, tmp_dir_);
                sfm_time = timer.Elapsed();
            }

            std::chrono::duration<double> mesh_recon_init_time;
            std::chrono::duration<double> mesh_recon_time;
            MeshReconstruction::Result mesh_recon_result;
            {
                timer.Restart();
                MeshReconstruction mesh_recon(exe_dir_);
                mesh_recon_init_time = timer.Elapsed();
                mesh_recon_result = mesh_recon.Process(sfm_result, tmp_dir_);
                mesh_recon_time = timer.Elapsed();
            }

            std::chrono::duration<double> mesh_gen_init_time;
            std::chrono::duration<double> mesh_gen_time;
            Mesh result_mesh;
            {
                timer.Restart();
                MeshGenerator mesh_gen(gpu_system_, python_system_);
                mesh_gen_init_time = timer.Elapsed();
                result_mesh = mesh_gen.Generate(sfm_result, mesh_recon_result, 2048, tmp_dir_);
                mesh_gen_time = timer.Elapsed();
            }

            std::cout << std::format("Structure from motion time: {:.3f} s (init {:.3f} s)\n", sfm_time.count(), sfm_init_time.count());
            std::cout << std::format(
                "Mesh reconstruction time: {:.3f} s (init {:.3f} s)\n", mesh_recon_time.count(), mesh_recon_init_time.count());
            std::cout << std::format("Mesh generation time: {:.3f} s (init {:.3f} s)\n", mesh_gen_time.count(), mesh_gen_init_time.count());

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
