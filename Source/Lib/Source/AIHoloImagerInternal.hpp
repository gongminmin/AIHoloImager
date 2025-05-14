// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <filesystem>
#include <string_view>

#include "Gpu/GpuSystem.hpp"
#include "Python/PythonSystem.hpp"

namespace AIHoloImager
{
    class AIHoloImagerInternal
    {
    public:
        virtual ~AIHoloImagerInternal() noexcept;

        virtual const std::filesystem::path& ExeDir() = 0;

        virtual GpuSystem& GpuSystemInstance() = 0;
        virtual PythonSystem& PythonSystemInstance() = 0;

        virtual void AddTiming(std::string_view name, std::chrono::duration<double> duration) = 0;
    };
} // namespace AIHoloImager
