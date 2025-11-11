// Copyright (c) 2025 Minmin Gong
//

#include "GpuSystemInternalFactory.hpp"

#include "D3D12/D3D12System.hpp"

namespace AIHoloImager
{
    std::unique_ptr<GpuSystemInternal> CreateGpuSystemInternal(GpuSystem::Api api, GpuSystem& gpu_system,
        std::function<bool(GpuSystem::Api api, void* device)> confirm_device, bool enable_sharing, bool enable_debug)
    {
        switch (api)
        {
        case GpuSystem::Api::D3D12:
            return std::make_unique<D3D12System>(gpu_system, std::move(confirm_device), enable_sharing, enable_debug);

        default:
            Unreachable("Invalid API");
        }
    }
} // namespace AIHoloImager
