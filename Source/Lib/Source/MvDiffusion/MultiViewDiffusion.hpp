// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include "AIHoloImager/Texture.hpp"
#include "Python/PythonSystem.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class MultiViewDiffusion
    {
        DISALLOW_COPY_AND_ASSIGN(MultiViewDiffusion);

    public:
        explicit MultiViewDiffusion(PythonSystem& python_system);
        MultiViewDiffusion(MultiViewDiffusion&& other) noexcept;
        ~MultiViewDiffusion() noexcept;

        MultiViewDiffusion& operator=(MultiViewDiffusion&& other) noexcept;

        Texture Generate(const Texture& input_image, uint32_t num_steps = 75);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
