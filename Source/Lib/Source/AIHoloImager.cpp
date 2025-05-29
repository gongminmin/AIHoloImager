// Copyright (c) 2024-2025 Minmin Gong
//

#include "AIHoloImager/AIHoloImager.hpp"

#include <chrono>
#include <format>
#include <iostream>
#include <mutex>

#include "AIHoloImagerInternal.hpp"
#include "Base/MiniWindows.hpp"
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
            : exe_dir_(RetrieveExeDir()), tmp_dir_(tmp_dir), gpu_system_(ConfirmDevice, true), python_system_(enable_cuda, exe_dir_)
        {
        }

        static bool ConfirmDevice(ID3D12Device* device)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
            device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1));
            if (!options1.WaveOps || (options1.WaveLaneCountMin < 16))
            {
                return false;
            }

            return true;
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
            std::lock_guard lock(timing_mutex_);
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
        std::mutex timing_mutex_;
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
