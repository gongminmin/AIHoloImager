// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>

#include "AIHoloImager/Mesh.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Python/PythonSystem.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class MultiViewRenderer
    {
        DISALLOW_COPY_AND_ASSIGN(MultiViewRenderer);

    public:
        struct Result
        {
            Texture multi_view_images[6]; // All in 3 channels
        };

    public:
        MultiViewRenderer(GpuSystem& gpu_system, PythonSystem& python_system, uint32_t width, uint32_t height);
        MultiViewRenderer(MultiViewRenderer&& other) noexcept;
        ~MultiViewRenderer() noexcept;

        MultiViewRenderer& operator=(MultiViewRenderer&& other) noexcept;

        Result Render(const Mesh& mesh);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
