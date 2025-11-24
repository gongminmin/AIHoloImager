// Copyright (c) 2024-2025 Minmin Gong
//

#include "AIHoloImager/AIHoloImager.hpp"

#include <chrono>
#include <format>
#include <iostream>

#ifdef AIHI_ENABLE_D3D12
    #include "Base/MiniWindows.hpp"

    #include <directx/d3d12.h>
#endif
#ifdef AIHI_ENABLE_VULKAN
    #include <volk.h>
#endif

#include "AIHoloImagerInternal.hpp"
#include "Base/ErrorHandling.hpp"
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
        Impl(DeviceType device, Api api, const std::filesystem::path& tmp_dir, bool gpu_debug)
            : exe_dir_(RetrieveExeDir()), tmp_dir_(tmp_dir), gpu_system_(Convert(api), ConfirmDevice, true, gpu_debug),
              python_system_(GetDeviceName(device), exe_dir_), tensor_converter_(gpu_system_, GetDeviceName(device))
        {
        }

        static std::filesystem::path RetrieveExeDir()
        {
            char exe_path[MAX_PATH];
            GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
            return std::filesystem::path(exe_path).parent_path();
        }

        static GpuSystem::Api Convert(AIHoloImager::Api api)
        {
            switch (api)
            {
#ifdef AIHI_ENABLE_D3D12
            case AIHoloImager::Api::D3D12:
                return GpuSystem::Api::D3D12;
#endif
#ifdef AIHI_ENABLE_VULKAN
            case AIHoloImager::Api::Vulkan:
                return GpuSystem::Api::Vulkan;
#endif

            case AIHoloImager::Api::Auto:
                return GpuSystem::Api::Auto;

            default:
                return GpuSystem::Api::Auto;
            }
        }

        static bool ConfirmDevice(GpuSystem::Api api, void* device)
        {
            switch (api)
            {
#ifdef AIHI_ENABLE_D3D12
            case GpuSystem::Api::D3D12:
            {
                D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
                reinterpret_cast<ID3D12Device*>(device)->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1));
                if (!options1.WaveOps || (options1.WaveLaneCountMin < 16))
                {
                    return false;
                }

                return true;
            }
#endif
#ifdef AIHI_ENABLE_VULKAN
            case GpuSystem::Api::Vulkan:
            {
                VkPhysicalDeviceSubgroupProperties subgroup_properties{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                };

                VkPhysicalDeviceProperties2 device_properties{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                    .pNext = &subgroup_properties,
                };

                vkGetPhysicalDeviceProperties2(reinterpret_cast<VkPhysicalDevice>(device), &device_properties);

                if (((subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) == 0) ||
                    (subgroup_properties.subgroupSize < 16))
                {
                    return false;
                }

                return true;
            }
#endif

            default:
                Unreachable("Invalid API");
            }
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

    AIHoloImager::AIHoloImager(DeviceType device, Api api, const std::filesystem::path& tmp_dir, bool gpu_debug)
        : impl_(std::make_unique<Impl>(device, api, tmp_dir, gpu_debug))
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
