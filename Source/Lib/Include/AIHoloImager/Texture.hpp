// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

#include "AIHoloImager/Common.hpp"

namespace AIHoloImager
{
    class Texture
    {
    public:
        AIHI_API Texture();
        AIHI_API Texture(uint32_t width, uint32_t height, uint32_t num_channels);
        AIHI_API Texture(const Texture& rhs);
        AIHI_API Texture(Texture&& rhs) noexcept;
        AIHI_API ~Texture() noexcept;

        AIHI_API Texture& operator=(const Texture& rhs);
        AIHI_API Texture& operator=(Texture&& rhs) noexcept;

        AIHI_API bool Valid() const noexcept;

        AIHI_API uint32_t Width() const noexcept;
        AIHI_API uint32_t Height() const noexcept;
        AIHI_API uint32_t NumChannels() const noexcept;

        AIHI_API uint8_t* Data() noexcept;
        AIHI_API const uint8_t* Data() const noexcept;
        AIHI_API uint32_t DataSize() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    AIHI_API Texture LoadTexture(const std::filesystem::path& path);
    AIHI_API void SaveTexture(const Texture& tex, const std::filesystem::path& path);
} // namespace AIHoloImager
