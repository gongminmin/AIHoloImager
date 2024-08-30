// Copyright (c) 2024 Minmin Gong
//

#include "MaskGenerator.hpp"

#include <cstddef>
#include <span>

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
        }

        Texture Generate(const Texture& input_image)
        {
            auto args = python_system_.MakeTuple(4);
            {
                auto image = python_system_.MakeObject(
                    std::span<const std::byte>(reinterpret_cast<const std::byte*>(input_image.Data()), input_image.DataSize()));
                python_system_.SetTupleItem(*args, 0, std::move(image));

                python_system_.SetTupleItem(*args, 1, python_system_.MakeObject(input_image.Width()));
                python_system_.SetTupleItem(*args, 2, python_system_.MakeObject(input_image.Height()));
                python_system_.SetTupleItem(*args, 3, python_system_.MakeObject(input_image.NumChannels()));
            }

            const auto mask_data = python_system_.CallObject(*mask_generator_gen_method_, *args);

            Texture mask_image(input_image.Width(), input_image.Height(), 1);
            const auto mask_span = python_system_.ToBytes(*mask_data);
            assert(mask_span.size() == mask_image.DataSize());
            std::memcpy(mask_image.Data(), mask_span.data(), mask_span.size());

            return mask_image;
        }

    private:
        PythonSystem& python_system_;

        PyObjectPtr mask_generator_module_;
        PyObjectPtr mask_generator_class_;
        PyObjectPtr mask_generator_;
        PyObjectPtr mask_generator_gen_method_;
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
