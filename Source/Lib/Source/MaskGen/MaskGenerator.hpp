// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>

#include "AIHoloImager/Texture.hpp"

namespace AIHoloImager
{
    class MaskGenerator
    {
    public:
        explicit MaskGenerator(const std::filesystem::path& exe_dir);
        MaskGenerator(const MaskGenerator& other) = delete;
        MaskGenerator(MaskGenerator&& other) noexcept;
        ~MaskGenerator() noexcept;

        MaskGenerator& operator=(const MaskGenerator& other) = delete;
        MaskGenerator& operator=(MaskGenerator&& other) noexcept;

        Texture Generate(const Texture& input_image);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
