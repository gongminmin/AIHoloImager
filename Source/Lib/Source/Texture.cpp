// Copyright (c) 2024 Minmin Gong
//

#include "AIHoloImager/Texture.hpp"

#include <cstring>
#include <format>
#include <fstream>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "Util/ErrorHandling.hpp"

namespace AIHoloImager
{
    class Texture::Impl
    {
    public:
        Impl(uint32_t width, uint32_t height, ElementFormat format)
            : width_(width), height_(height), format_(format), data_(width * height * FormatSize(format))
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
        ElementFormat Format() const noexcept
        {
            return format_;
        }

        std::byte* Data() noexcept
        {
            return data_.data();
        }
        const std::byte* Data() const noexcept
        {
            return data_.data();
        }

        uint32_t DataSize() const noexcept
        {
            return static_cast<uint32_t>(data_.size());
        }

        Texture Convert(ElementFormat format) const
        {
            Texture ret(width_, height_, format);
            this->ConvertData(ret.impl_->data_, format);
            return ret;
        }

        void ConvertInPlace(ElementFormat format)
        {
            std::vector<std::byte> new_data;
            this->ConvertData(new_data, format);
            data_ = std::move(new_data);
            format_ = format;
        }

    private:
        void ConvertData(std::vector<std::byte>& dst_data, ElementFormat format) const
        {
            dst_data.resize(width_ * height_ * FormatSize(format));

            const std::byte* src = data_.data();
            std::byte* dst = dst_data.data();

            if (format == format_)
            {
                std::memcpy(dst, src, data_.size());
            }
            else
            {
                if (FormatChannelSize(format_) == 1)
                {
                    const uint32_t channels = FormatChannels(format_);

                    if ((format == ElementFormat::RGBA8_UNorm))
                    {
#ifdef _OPENMP
    #pragma omp parallel
#endif
                        for (uint32_t i = 0; i < width_ * height_; ++i)
                        {
                            memcpy(&dst[i * 4], &src[i * channels], channels);
                            dst[i * 4 + 3] = std::byte(0xFF);
                        }

                        return;
                    }
                    else if (format == ElementFormat::RGB8_UNorm)
                    {
                        const uint32_t copy_channels = std::min(channels, 3u);

#ifdef _OPENMP
    #pragma omp parallel
#endif
                        for (uint32_t i = 0; i < width_ * height_; ++i)
                        {
                            memcpy(&dst[i * 3], &src[i * channels], copy_channels);
                            if (copy_channels < 3)
                            {
                                memset(&dst[i * 3 + copy_channels], 0, 3 - copy_channels);
                            }
                        }

                        return;
                    }
                }
                else if ((format == ElementFormat::R32_Float) || (format == ElementFormat::RGB32_Float))
                {
                    const uint32_t src_channels = FormatChannels(format_);
                    const uint32_t dst_channels = FormatChannels(format);
                    const uint32_t ch_size = FormatChannelSize(format_);

                    const float* src_float = reinterpret_cast<const float*>(data_.data());
                    float* dst_float = reinterpret_cast<float*>(dst_data.data());

                    const uint32_t copy_channels = std::min(src_channels, dst_channels);

#ifdef _OPENMP
    #pragma omp parallel
#endif
                    for (uint32_t i = 0; i < width_ * height_; ++i)
                    {
                        memcpy(&dst_float[i * dst_channels], &src_float[i * src_channels], copy_channels * ch_size);
                        if (copy_channels < 3)
                        {
                            memset(&dst[i * dst_channels + copy_channels], 0, (dst_channels - copy_channels) * ch_size);
                        }
                    }

                    return;
                }

                // TODO: Support more formats
                Unreachable("Unsupported conversion");
            }
        }

    private:
        uint32_t width_ = 0;
        uint32_t height_ = 0;
        ElementFormat format_ = ElementFormat::Unknown;

        std::vector<std::byte> data_;
    };

    Texture::Texture() = default;
    Texture::Texture(uint32_t width, uint32_t height, ElementFormat format) : impl_(std::make_unique<Impl>(width, height, format))
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
    ElementFormat Texture::Format() const noexcept
    {
        return impl_->Format();
    }

    std::byte* Texture::Data() noexcept
    {
        return impl_->Data();
    }
    const std::byte* Texture::Data() const noexcept
    {
        return impl_->Data();
    }

    uint32_t Texture::DataSize() const noexcept
    {
        return impl_->DataSize();
    }

    Texture Texture::Convert(ElementFormat format) const
    {
        return impl_->Convert(format);
    }

    void Texture::ConvertInPlace(ElementFormat format)
    {
        impl_->ConvertInPlace(format);
    }

    Texture LoadTexture(const std::filesystem::path& path)
    {
        int width, height;
        uint8_t* data = stbi_load(path.string().c_str(), &width, &height, nullptr, 4);

        Texture tex(width, height, ElementFormat::RGBA8_UNorm);
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

        const ElementFormat format = tex.Format();
        const uint32_t num_channels = FormatChannels(format);

        const auto output_ext = path.extension();
        if (FormatChannelSize(format) == 1)
        {
            if (output_ext == ".jpg")
            {
                stbi_write_jpg(
                    path.string().c_str(), static_cast<int>(tex.Width()), static_cast<int>(tex.Height()), num_channels, tex.Data(), 90);
                return;
            }
            else if (output_ext == ".png")
            {
                stbi_write_png(path.string().c_str(), static_cast<int>(tex.Width()), static_cast<int>(tex.Height()), num_channels,
                    tex.Data(), static_cast<int>(tex.Width() * num_channels));
                return;
            }
        }
        else if ((format == ElementFormat::R32_Float) || (format == ElementFormat::RGB32_Float))
        {
            if (output_ext == ".pfm")
            {
                std::ofstream pfm_fs(path, std::ios_base::binary);
                const std::string header =
                    std::format("P{}\n{} {}\n-1.0\n", format == ElementFormat::R32_Float ? 'f' : 'F', tex.Width(), tex.Height());
                pfm_fs.write(header.c_str(), header.size());

                const uint32_t row_size = tex.Width() * FormatSize(format);
                const std::byte* data = tex.Data() + (tex.Height() - 1) * row_size;
                for (uint32_t y = 0; y < tex.Height(); ++y)
                {
                    pfm_fs.write(reinterpret_cast<const char*>(data), row_size);
                    data -= row_size;
                }

                return;
            }
        }

        Unreachable("Unsupported format");
    }
} // namespace AIHoloImager
