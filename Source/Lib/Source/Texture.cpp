// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/Texture.hpp"

#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace AIHoloImager
{
    class Texture::Impl
    {
    public:
        Impl(uint32_t width, uint32_t height, uint32_t num_channels)
            : width_(width), height_(height), num_channels_(num_channels), data_(width * height * num_channels)
        {
        }

        uint32_t Width() const noexcept
        {
            return width_;
        }
        uint32_t Height() const noexcept
        {
            return height_;
        }
        uint32_t NumChannels() const noexcept
        {
            return num_channels_;
        }

        uint8_t* Data() noexcept
        {
            return data_.data();
        }
        const uint8_t* Data() const noexcept
        {
            return data_.data();
        }

        uint32_t DataSize() const noexcept
        {
            return static_cast<uint32_t>(data_.size());
        }

    private:
        uint32_t width_ = 0;
        uint32_t height_ = 0;
        uint32_t num_channels_ = 0;

        std::vector<uint8_t> data_;
    };

    Texture::Texture() = default;
    Texture::Texture(uint32_t width, uint32_t height, uint32_t num_channels) : impl_(std::make_unique<Impl>(width, height, num_channels))
    {
    }
    Texture::Texture(const Texture& rhs) : impl_(rhs.impl_ ? std::make_unique<Impl>(*rhs.impl_) : nullptr)
    {
    }
    Texture::Texture(Texture&& rhs) noexcept = default;
    Texture::~Texture() noexcept = default;

    Texture& Texture::operator=(const Texture& rhs)
    {
        if (this != &rhs)
        {
            if (rhs.impl_)
            {
                if (impl_)
                {
                    *impl_ = *rhs.impl_;
                }
                else
                {
                    impl_ = std::make_unique<Impl>(*rhs.impl_);
                }
            }
            else
            {
                impl_.reset();
            }
        }
        return *this;
    }
    Texture& Texture::operator=(Texture&& rhs) noexcept = default;

    bool Texture::Valid() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    uint32_t Texture::Width() const noexcept
    {
        return impl_->Width();
    }
    uint32_t Texture::Height() const noexcept
    {
        return impl_->Height();
    }
    uint32_t Texture::NumChannels() const noexcept
    {
        return impl_->NumChannels();
    }

    uint8_t* Texture::Data() noexcept
    {
        return impl_->Data();
    }
    const uint8_t* Texture::Data() const noexcept
    {
        return impl_->Data();
    }

    uint32_t Texture::DataSize() const noexcept
    {
        return impl_->DataSize();
    }

    Texture LoadTexture(const std::filesystem::path& path)
    {
        int width, height;
        uint8_t* data = stbi_load(path.string().c_str(), &width, &height, nullptr, 4);

        Texture tex(width, height, 4);
        if (data != nullptr)
        {
            std::memcpy(tex.Data(), data, tex.DataSize());
            stbi_image_free(data);
        }

        return tex;
    }

    void SaveTexture(const Texture& tex, const std::filesystem::path& path)
    {
        if (!tex.Valid())
        {
            return;
        }

        const auto output_ext = path.extension();
        if (output_ext == ".jpg")
        {
            stbi_write_jpg(
                path.string().c_str(), static_cast<int>(tex.Width()), static_cast<int>(tex.Height()), tex.NumChannels(), tex.Data(), 90);
        }
        else
        {
            stbi_write_png(path.string().c_str(), static_cast<int>(tex.Width()), static_cast<int>(tex.Height()), tex.NumChannels(),
                tex.Data(), static_cast<int>(tex.Width() * tex.NumChannels()));
        }
    }

    void Ensure4Channel(Texture& tex)
    {
        const uint32_t channels = tex.NumChannels();
        if (channels != 4)
        {
            Texture ret(tex.Width(), tex.Height(), 4);

            const uint8_t* src = tex.Data();
            uint8_t* dst = ret.Data();

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < tex.Width() * tex.Height(); ++i)
            {
                memcpy(&dst[i * 4], &src[i * channels], channels);
                dst[i * 4 + 3] = 0xFF;
            }

            tex = std::move(ret);
        }
    }

    void RemoveAlpha(Texture& tex)
    {
        const uint32_t channels = tex.NumChannels();
        if (channels != 3)
        {
            Texture ret(tex.Width(), tex.Height(), 3);

            const uint8_t* src = tex.Data();
            uint8_t* dst = ret.Data();

#ifdef _OPENMP
    #pragma omp parallel
#endif
            for (uint32_t i = 0; i < tex.Width() * tex.Height(); ++i)
            {
                memcpy(&dst[i * 3], &src[i * channels], 3);
            }

            tex = std::move(ret);
        }
    }
} // namespace AIHoloImager
