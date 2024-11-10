// Copyright (c) 2024 Minmin Gong
//

#include "MaskGenerator.hpp"

#include <cstddef>
#include <span>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>

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
            mask_generator_set_image_method_ = python_system_.GetAttr(*mask_generator_, "SetImage");
            mask_generator_gen_method_ = python_system_.GetAttr(*mask_generator_, "Gen");
        }

        ~Impl()
        {
            auto mask_generator_destroy_method = python_system_.GetAttr(*mask_generator_, "Destroy");
            python_system_.CallObject(*mask_generator_destroy_method);
        }

        void Generate(Texture& image)
        {
            Ensure4Channel(image);

            const uint32_t width = image.Width();
            const uint32_t height = image.Height();
            const uint32_t channels = image.NumChannels();

            {
                auto args = python_system_.MakeTuple(4);
                {
                    auto py_image = python_system_.MakeObject(
                        std::span<const std::byte>(reinterpret_cast<const std::byte*>(image.Data()), width * height * channels));
                    python_system_.SetTupleItem(*args, 0, std::move(py_image));

                    python_system_.SetTupleItem(*args, 1, python_system_.MakeObject(width));
                    python_system_.SetTupleItem(*args, 2, python_system_.MakeObject(height));
                    python_system_.SetTupleItem(*args, 3, python_system_.MakeObject(channels));
                }

                python_system_.CallObject(*mask_generator_set_image_method_, *args);
            }

            auto mask = this->GenMask(0, 0, width, height);
            auto mask_span = python_system_.ToBytes(*mask);
            assert(mask_span.size() == width * height);

            const uint8_t* mask_data = reinterpret_cast<const uint8_t*>(mask_span.data());
            if ((width > 320) || (height > 320))
            {
                glm::uvec2 bb_min(width, height);
                glm::uvec2 bb_max(0, 0);
                for (uint32_t y = 0; y < height; ++y)
                {
                    for (uint32_t x = 0; x < width; ++x)
                    {
                        if (mask_data[y * width + x] > 127)
                        {
                            const glm::uvec2 coord = glm::uvec2(x, y);
                            bb_min = glm::min(bb_min, coord);
                            bb_max = glm::max(bb_max, coord);
                        }
                    }
                }

                const uint32_t crop_extent = std::max(std::max(bb_max.x - bb_min.x, bb_max.y - bb_min.y) + 8, 320U) / 2u;
                const glm::uvec2 crop_center = (bb_min + bb_max) / 2u;
                bb_min = glm::clamp(glm::ivec2(crop_center - crop_extent), glm::ivec2(0, 0), glm::ivec2(width, height));
                bb_max = glm::clamp(glm::ivec2(crop_center + crop_extent), glm::ivec2(0, 0), glm::ivec2(width, height));

                mask = this->GenMask(bb_min.x, bb_min.y, bb_max.x + 1, bb_max.y + 1);
                mask_span = python_system_.ToBytes(*mask);
                mask_data = reinterpret_cast<const uint8_t*>(mask_span.data());
            }

            uint8_t* image_data = image.Data();
            for (uint32_t i = 0; i < width * height; ++i)
            {
                image_data[i * 4 + 3] = mask_data[i];
            }
        }

    private:
        PyObjectPtr GenMask(uint32_t roi_left, uint32_t roi_top, uint32_t roi_right, uint32_t roi_bottom)
        {
            auto args = python_system_.MakeTuple(4);
            {
                python_system_.SetTupleItem(*args, 0, python_system_.MakeObject(roi_left));
                python_system_.SetTupleItem(*args, 1, python_system_.MakeObject(roi_top));
                python_system_.SetTupleItem(*args, 2, python_system_.MakeObject(roi_right));
                python_system_.SetTupleItem(*args, 3, python_system_.MakeObject(roi_bottom));
            }

            return python_system_.CallObject(*mask_generator_gen_method_, *args);
        }

    private:
        PythonSystem& python_system_;

        PyObjectPtr mask_generator_module_;
        PyObjectPtr mask_generator_class_;
        PyObjectPtr mask_generator_;
        PyObjectPtr mask_generator_set_image_method_;
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
