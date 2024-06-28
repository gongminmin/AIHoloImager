// Copyright (c) 2024 Minmin Gong
//

#include "MaskGenerator.hpp"

#include <cstddef>
#include <span>
#include <string_view>

namespace AIHoloImager
{
    class MaskGenerator::Impl
    {
    public:
        explicit Impl(PythonSystem& python_system) : python_system_(python_system)
        {
            mask_generator_module_ = python_system_.Import("MaskGenerator");
            mask_generator_class_ = python_system_.GetAttr(*mask_generator_module_, "MaskGenerator");
            mask_generator_ = python_system_.CallObject(*mask_generator_class_);
            mask_generator_gen_method_ = python_system_.GetAttr(*mask_generator_, "Gen");

            pil_module_ = python_system_.Import("PIL");
            image_class_ = python_system_.GetAttr(*pil_module_, "Image");
            image_frombuffer_method_ = python_system_.GetAttr(*image_class_, "frombuffer");
        }

        Texture Generate(const Texture& input_image)
        {
            PyObjectPtr py_input_image;
            {
                auto args = python_system_.MakeTuple(3);
                {
                    std::wstring_view mode;
                    switch (input_image.NumChannels())
                    {
                    case 1:
                        mode = L"L";
                        break;
                    case 3:
                        mode = L"RGB";
                        break;
                    case 4:
                        mode = L"RGBA";
                        break;
                    }

                    python_system_.SetTupleItem(*args, 0, python_system_.MakeObject(mode));
                }
                {
                    auto size = python_system_.MakeTuple(2);
                    {
                        python_system_.SetTupleItem(*size, 0, python_system_.MakeObject(input_image.Width()));
                        python_system_.SetTupleItem(*size, 1, python_system_.MakeObject(input_image.Height()));
                    }
                    python_system_.SetTupleItem(*args, 1, std::move(size));
                }
                {
                    auto image = python_system_.MakeObject(
                        std::span<const std::byte>(reinterpret_cast<const std::byte*>(input_image.Data()), input_image.DataSize()));
                    python_system_.SetTupleItem(*args, 2, std::move(image));
                }

                py_input_image = python_system_.CallObject(*image_frombuffer_method_, *args);
            }

            auto args = python_system_.MakeTuple(1);
            {
                python_system_.SetTupleItem(*args, 0, *py_input_image);
            }

            auto py_mask_image = python_system_.CallObject(*mask_generator_gen_method_, *args);
            auto tobytes_method = python_system_.GetAttr(*py_mask_image, "tobytes");
            auto mask_data = python_system_.CallObject(*tobytes_method);

            const uint32_t width = python_system_.GetAttrOfType<long>(*py_mask_image, "width");
            const uint32_t height = python_system_.GetAttrOfType<long>(*py_mask_image, "height");
            uint32_t num_channels = 3;
            const std::wstring_view mode_str = python_system_.GetAttrOfType<std::wstring_view>(*py_mask_image, "mode");
            if (mode_str == L"L")
            {
                num_channels = 1;
            }
            else if (mode_str == L"RGB")
            {
                num_channels = 3;
            }
            else if ((mode_str == L"RGBA") || (mode_str == L"RGBX"))
            {
                num_channels = 4;
            }

            Texture mask_image(width, height, num_channels);
            std::memcpy(mask_image.Data(), python_system_.Cast<std::span<const std::byte>>(*mask_data).data(), mask_image.DataSize());

            return mask_image;
        }

    private:
        PythonSystem& python_system_;

        PyObjectPtr mask_generator_module_;
        PyObjectPtr mask_generator_class_;
        PyObjectPtr mask_generator_;
        PyObjectPtr mask_generator_gen_method_;

        PyObjectPtr pil_module_;
        PyObjectPtr image_class_;
        PyObjectPtr image_frombuffer_method_;
    };

    MaskGenerator::MaskGenerator(PythonSystem& python_system) : impl_(std::make_unique<Impl>(python_system))
    {
    }

    MaskGenerator::~MaskGenerator() noexcept = default;

    MaskGenerator::MaskGenerator(MaskGenerator&& other) noexcept = default;
    MaskGenerator& MaskGenerator::operator=(MaskGenerator&& other) noexcept = default;

    Texture MaskGenerator::Generate(const Texture& input_image)
    {
        return impl_->Generate(input_image);
    }
} // namespace AIHoloImager
