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

        void Generate(Texture& image)
        {
            auto args = python_system_.MakeTuple(4);
            {
                auto py_image = python_system_.MakeObject(
                    std::span<const std::byte>(reinterpret_cast<const std::byte*>(image.Data()), image.DataSize()));
                python_system_.SetTupleItem(*args, 0, std::move(py_image));

                python_system_.SetTupleItem(*args, 1, python_system_.MakeObject(image.Width()));
                python_system_.SetTupleItem(*args, 2, python_system_.MakeObject(image.Height()));
                python_system_.SetTupleItem(*args, 3, python_system_.MakeObject(image.NumChannels()));
            }

            Ensure4Channel(image);

            const auto mask_data = python_system_.CallObject(*mask_generator_gen_method_, *args);

            const auto mask_span = python_system_.ToBytes(*mask_data);
            assert(mask_span.size() == image.Width() * image.Height());
            const uint8_t* src = reinterpret_cast<const uint8_t*>(mask_span.data());
            uint8_t* dst = &image.Data()[3];
            for (uint32_t i = 0; i < mask_span.size(); ++i)
            {
                dst[i * 4] = src[i];
            }
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

    void MaskGenerator::Generate(Texture& image)
    {
        impl_->Generate(image);
    }
} // namespace AIHoloImager
