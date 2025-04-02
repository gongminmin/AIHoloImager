// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <glm/vec4.hpp>

#include "Base/Noncopyable.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"
#include "Python/PythonSystem.hpp"

namespace AIHoloImager
{
    class MaskGenerator
    {
        DISALLOW_COPY_AND_ASSIGN(MaskGenerator);

    public:
        MaskGenerator(GpuSystem& gpu_system, PythonSystem& python_system);
        MaskGenerator(MaskGenerator&& other) noexcept;
        ~MaskGenerator() noexcept;

        MaskGenerator& operator=(MaskGenerator&& other) noexcept;

        void Generate(GpuCommandList& cmd_list, GpuTexture2D& image, glm::uvec4& roi);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
