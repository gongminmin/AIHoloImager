// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <filesystem>

#include "Gpu/GpuSystem.hpp"
#include "GpuGSplatDiffRender/GpuGSplatDiffRender.hpp"

namespace AIHoloImager
{
    void SavePointCloud(GpuSystem& gpu_system, const Gaussians& gaussians, const std::filesystem::path& path);

    Gaussians LoadGaussians(GpuSystem& gpu_system, const std::filesystem::path& path);
    void SaveGaussians(GpuSystem& gpu_system, const Gaussians& gaussians, const std::filesystem::path& path);
} // namespace AIHoloImager
