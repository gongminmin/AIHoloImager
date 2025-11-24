// Copyright (c) 2025 Minmin Gong
//

#include "GpuSystemInternalFactory.hpp"

#include "Base/ErrorHandling.hpp"

#ifdef AIHI_ENABLE_D3D12
    #include "D3D12/D3D12System.hpp"
#endif
#ifdef AIHI_ENABLE_VULKAN
    #include "Vulkan/VulkanSystem.hpp"
#endif

namespace AIHoloImager
{
    std::unique_ptr<GpuSystemInternal> CreateGpuSystemInternal(GpuSystem::Api api, GpuSystem& gpu_system,
        std::function<bool(GpuSystem::Api api, void* device)> confirm_device, bool enable_sharing, bool enable_debug)
    {
        switch (api)
        {
#ifdef AIHI_ENABLE_D3D12
        case GpuSystem::Api::D3D12:
            return std::make_unique<D3D12System>(gpu_system, std::move(confirm_device), enable_sharing, enable_debug);
#endif
#ifdef AIHI_ENABLE_VULKAN
        case GpuSystem::Api::Vulkan:
            return std::make_unique<VulkanSystem>(gpu_system, std::move(confirm_device), enable_sharing, enable_debug);
#endif

        default:
            Unreachable("Invalid API");
        }
    }
} // namespace AIHoloImager
