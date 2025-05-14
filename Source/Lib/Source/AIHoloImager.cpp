// Copyright (c) 2024-2025 Minmin Gong
//

#include "AIHoloImager/AIHoloImager.hpp"

#include <chrono>
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

#include "AIHoloImagerInternal.hpp"
#include "Gpu/GpuSystem.hpp"
#include "MeshGen/MeshGenerator.hpp"
#include "Python/PythonSystem.hpp"
#include "SfM/StructureFromMotion.hpp"

namespace
{
    std::filesystem::path RetrieveExeDir()
    {
        char exe_path[MAX_PATH];
        GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
        return std::filesystem::path(exe_path).parent_path();
    }
} // namespace

namespace AIHoloImager
{
    AIHoloImagerInternal::~AIHoloImagerInternal() noexcept = default;

    class AIHoloImager::Impl : public AIHoloImagerInternal
    {
    public:
        Impl(bool enable_cuda, const std::filesystem::path& tmp_dir)
            : exe_dir_(RetrieveExeDir()), tmp_dir_(tmp_dir), gpu_system_(nullptr, true), python_system_(enable_cuda, exe_dir_)
        {
        }

        const std::filesystem::path& ExeDir() override
        {
            return exe_dir_;
        }

        GpuSystem& GpuSystemInstance() override
        {
            return gpu_system_;
        }

        PythonSystem& PythonSystemInstance() override
        {
            return python_system_;
        }

        void AddTiming(std::string_view name, std::chrono::duration<double> duration)
        {
            timings_.push_back(std::make_tuple(std::string(name), std::move(duration)));
        }

        Mesh Generate(const std::filesystem::path& input_path)
        {
            StructureFromMotion::Result sfm_result;
            {
                StructureFromMotion sfm(*this);
                sfm_result = sfm.Process(input_path, true, tmp_dir_);
            }

            Mesh result_mesh;
            {
                MeshGenerator mesh_gen(*this);
                result_mesh = mesh_gen.Generate(sfm_result, 2048, tmp_dir_);
            }

            for (auto& [name, duration] : timings_)
            {
                std::cout << std::format("{}: {:.3f} s\n", name, duration.count());
            }

            return result_mesh;
        }

    private:
        std::filesystem::path exe_dir_;
        std::filesystem::path tmp_dir_;

        GpuSystem gpu_system_;
        PythonSystem python_system_;

        std::vector<std::tuple<std::string, std::chrono::duration<double>>> timings_;
    };

    AIHoloImager::AIHoloImager(bool enable_cuda, const std::filesystem::path& tmp_dir) : impl_(std::make_unique<Impl>(enable_cuda, tmp_dir))
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
