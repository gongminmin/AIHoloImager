// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

namespace AIHoloImager
{
    class Texture
    {
    public:
        Texture();
        Texture(uint32_t width, uint32_t height, uint32_t num_channels);
        Texture(const Texture& rhs);
        Texture(Texture&& rhs) noexcept;
        ~Texture() noexcept;

        Texture& operator=(const Texture& rhs);
        Texture& operator=(Texture&& rhs) noexcept;

        bool Valid() const noexcept;

        uint32_t Width() const noexcept;
        uint32_t Height() const noexcept;
        uint32_t NumChannels() const noexcept;

        uint8_t* Data() noexcept;
        const uint8_t* Data() const noexcept;
        uint32_t DataSize() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    Texture LoadTexture(const std::filesystem::path& path);
    void SaveTexture(const Texture& tex, const std::filesystem::path& path);
} // namespace AIHoloImager
