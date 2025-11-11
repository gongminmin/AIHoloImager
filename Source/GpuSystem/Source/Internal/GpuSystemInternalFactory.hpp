// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <functional>
#include <memory>

#include "GpuSystemInternal.hpp"

namespace AIHoloImager
{
    std::unique_ptr<GpuSystemInternal> CreateGpuSystemInternal(GpuSystem::Api api, GpuSystem& gpu_system,
        std::function<bool(GpuSystem::Api api, void* device)> confirm_device = nullptr, bool enable_sharing = false,
        bool enable_debug = false);
} // namespace AIHoloImager
