// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>

#include "AIHoloImager/Texture.hpp"
#include "Util/Noncopyable.hpp"

namespace AIHoloImager
{
    class MaskGenerator
    {
        DISALLOW_COPY_AND_ASSIGN(MaskGenerator);

    public:
        explicit MaskGenerator(const std::filesystem::path& exe_dir);
        MaskGenerator(MaskGenerator&& other) noexcept;
        ~MaskGenerator() noexcept;

        MaskGenerator& operator=(MaskGenerator&& other) noexcept;

        Texture Generate(const Texture& input_image);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
