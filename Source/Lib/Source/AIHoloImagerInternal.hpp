// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"
#include "Python/PythonSystem.hpp"
#include "TensorConverter/TensorConverter.hpp"
#include "Util/PerfProfiler.hpp"

namespace AIHoloImager
{
    class AIHoloImagerInternal
    {
    public:
        struct ProjectionDesc
        {
            std::shared_ptr<GpuTexture2D> image;

            glm::mat4x4 view_mtx;
            glm::mat4x4 proj_mtx;
            uint32_t full_width = 0;
            uint32_t full_height = 0;
            glm::vec2 vp_offset = glm::vec2(0, 0);
            glm::uvec2 image_offset = glm::uvec2(0, 0);
        };

    public:
        virtual ~AIHoloImagerInternal() noexcept;

        virtual const std::filesystem::path& ExeDir() noexcept = 0;
        virtual const std::filesystem::path& TmpDir() noexcept = 0;

        virtual GpuSystem& GpuSystemInstance() noexcept = 0;
        virtual PythonSystem& PythonSystemInstance() noexcept = 0;
        virtual PerfProfiler& PerfProfilerInstance() noexcept = 0;

        virtual TensorConverter& TensorConverterInstance() noexcept = 0;
    };
} // namespace AIHoloImager
