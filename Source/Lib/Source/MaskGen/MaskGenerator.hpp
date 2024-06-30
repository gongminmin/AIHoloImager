// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include "AIHoloImager/Texture.hpp"
#include "Python/PythonSystem.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class MaskGenerator
    {
        DISALLOW_COPY_AND_ASSIGN(MaskGenerator);

    public:
        explicit MaskGenerator(PythonSystem& python_system);
        MaskGenerator(MaskGenerator&& other) noexcept;
        ~MaskGenerator() noexcept;

        MaskGenerator& operator=(MaskGenerator&& other) noexcept;

        Texture Generate(const Texture& input_image);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
