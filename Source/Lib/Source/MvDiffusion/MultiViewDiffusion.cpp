// Copyright (c) 2024 Minmin Gong
//

#include "MultiViewDiffusion.hpp"

#include <string_view>

namespace AIHoloImager
{
    class MultiViewDiffusion::Impl
    {
    public:
        explicit Impl(PythonSystem& python_system) : python_system_(python_system)
        {
            mv_diffusion_module_ = python_system_.Import("MultiViewDiffusion");
            mv_diffusion_class_ = python_system_.GetAttr(*mv_diffusion_module_, "MultiViewDiffusion");
            mv_diffusion_ = python_system_.CallObject(*mv_diffusion_class_);
            mv_diffusion_gen_method_ = python_system_.GetAttr(*mv_diffusion_, "Gen");
        }

        Texture Generate(const Texture& input_image, uint32_t num_steps)
        {
            auto args = python_system_.MakeTuple(5);
            {
                auto image = python_system_.MakeObject(
                    std::span<const std::byte>(reinterpret_cast<const std::byte*>(input_image.Data()), input_image.DataSize()));
                python_system_.SetTupleItem(*args, 0, std::move(image));

                python_system_.SetTupleItem(*args, 1, python_system_.MakeObject(input_image.Width()));
                python_system_.SetTupleItem(*args, 2, python_system_.MakeObject(input_image.Height()));
                python_system_.SetTupleItem(*args, 3, python_system_.MakeObject(input_image.NumChannels()));

                python_system_.SetTupleItem(*args, 4, python_system_.MakeObject(num_steps));
            }

            const auto mv_image_tuple = python_system_.CallObject(*mv_diffusion_gen_method_, *args);

            const auto mv_image_data = python_system_.GetTupleItem(*mv_image_tuple, 0);
            const uint32_t width = python_system_.Cast<long>(*python_system_.GetTupleItem(*mv_image_tuple, 1));
            const uint32_t height = python_system_.Cast<long>(*python_system_.GetTupleItem(*mv_image_tuple, 2));
            const uint32_t num_channels = python_system_.Cast<long>(*python_system_.GetTupleItem(*mv_image_tuple, 3));

            Texture mv_image(width, height, num_channels);
            const auto mv_image_span = python_system_.ToBytes(*mv_image_data);
            assert(mv_image_span.size() == mv_image.DataSize());
            std::memcpy(mv_image.Data(), mv_image_span.data(), mv_image_span.size());

            return mv_image;
        }

    private:
        PythonSystem& python_system_;

        PyObjectPtr mv_diffusion_module_;
        PyObjectPtr mv_diffusion_class_;
        PyObjectPtr mv_diffusion_;
        PyObjectPtr mv_diffusion_gen_method_;
    };

    MultiViewDiffusion::MultiViewDiffusion(PythonSystem& python_system) : impl_(std::make_unique<Impl>(python_system))
    {
    }

    MultiViewDiffusion::~MultiViewDiffusion() noexcept = default;

    MultiViewDiffusion::MultiViewDiffusion(MultiViewDiffusion&& other) noexcept = default;
    MultiViewDiffusion& MultiViewDiffusion::operator=(MultiViewDiffusion&& other) noexcept = default;

    Texture MultiViewDiffusion::Generate(const Texture& input_image, uint32_t num_steps)
    {
        return impl_->Generate(input_image, num_steps);
    }
} // namespace AIHoloImager
