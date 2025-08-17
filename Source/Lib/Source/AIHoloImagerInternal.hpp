// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <filesystem>

#include "Gpu/GpuSystem.hpp"
#include "Python/PythonSystem.hpp"
#include "Util/PerfProfiler.hpp"

namespace AIHoloImager
{
    class AIHoloImagerInternal
    {
    public:
        virtual ~AIHoloImagerInternal() noexcept;

        virtual const std::filesystem::path& ExeDir() noexcept = 0;
        virtual const std::filesystem::path& TmpDir() noexcept = 0;

        virtual GpuSystem& GpuSystemInstance() noexcept = 0;
        virtual PythonSystem& PythonSystemInstance() noexcept = 0;
        virtual PerfProfiler& PerfProfilerInstance() noexcept = 0;
    };
} // namespace AIHoloImager
