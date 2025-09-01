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
#include "Util/PerfProfiler.hpp"

namespace AIHoloImager
{
    AIHoloImagerInternal::~AIHoloImagerInternal() noexcept = default;

    class AIHoloImager::Impl : public AIHoloImagerInternal
    {
    public:
        Impl(DeviceType device, const std::filesystem::path& tmp_dir)
            : exe_dir_(RetrieveExeDir()), tmp_dir_(tmp_dir), gpu_system_(ConfirmDevice, true),
              python_system_(GetDeviceName(device), exe_dir_), tensor_converter_(gpu_system_, GetDeviceName(device))
        {
        }

        static std::filesystem::path RetrieveExeDir()
        {
            char exe_path[MAX_PATH];
            GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
            return std::filesystem::path(exe_path).parent_path();
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

        static std::string_view GetDeviceName(DeviceType device)
        {
            switch (device)
            {
            case DeviceType::Cuda:
                return "cuda";

            default:
                return "cpu";
            }
        }

        const std::filesystem::path& ExeDir() noexcept override
        {
            return exe_dir_;
        }

        const std::filesystem::path& TmpDir() noexcept override
        {
            return tmp_dir_;
        }

        GpuSystem& GpuSystemInstance() noexcept override
        {
            return gpu_system_;
        }

        PythonSystem& PythonSystemInstance() noexcept override
        {
            return python_system_;
        }

        PerfProfiler& PerfProfilerInstance() noexcept override
        {
            return profiler_;
        }

        TensorConverter& TensorConverterInstance() noexcept override
        {
            return tensor_converter_;
        }

        Mesh Generate(const std::filesystem::path& input_path, uint32_t texture_size, bool no_delight)
        {
            StructureFromMotion::Result sfm_result;
            {
                StructureFromMotion sfm(*this);
                sfm_result = sfm.Process(input_path, true, no_delight);
            }

            Mesh result_mesh;
            {
                MeshGenerator mesh_gen(*this);
                result_mesh = mesh_gen.Generate(sfm_result, texture_size);
            }

            profiler_.Output(std::cout);

            return result_mesh;
        }

    private:
        std::filesystem::path exe_dir_;
        std::filesystem::path tmp_dir_;

        GpuSystem gpu_system_;
        PythonSystem python_system_;
        PerfProfiler profiler_;

        TensorConverter tensor_converter_;
    };

    AIHoloImager::AIHoloImager(DeviceType device, const std::filesystem::path& tmp_dir) : impl_(std::make_unique<Impl>(device, tmp_dir))
    {
    }
    AIHoloImager::AIHoloImager(AIHoloImager&& rhs) noexcept = default;
    AIHoloImager::~AIHoloImager() noexcept = default;

    AIHoloImager& AIHoloImager::operator=(AIHoloImager&& rhs) noexcept = default;

    Mesh AIHoloImager::Generate(const std::filesystem::path& input_path, uint32_t texture_size, bool no_delight)
    {
        return impl_->Generate(input_path, texture_size, no_delight);
    }
} // namespace AIHoloImager
