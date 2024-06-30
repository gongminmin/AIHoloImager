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

            pil_module_ = python_system_.Import("PIL");
            image_class_ = python_system_.GetAttr(*pil_module_, "Image");
            image_frombuffer_method_ = python_system_.GetAttr(*image_class_, "frombuffer");
        }

        Texture Generate(const Texture& input_image, uint32_t num_steps)
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

            auto args = python_system_.MakeTuple(2);
            {
                python_system_.SetTupleItem(*args, 0, *py_input_image);
            }
            {
                python_system_.SetTupleItem(*args, 1, python_system_.MakeObject(num_steps));
            }

            auto py_mv_image = python_system_.CallObject(*mv_diffusion_gen_method_, *args);
            auto tobytes_method = python_system_.GetAttr(*py_mv_image, "tobytes");
            auto mask_data = python_system_.CallObject(*tobytes_method);

            const uint32_t width = python_system_.GetAttrOfType<long>(*py_mv_image, "width");
            const uint32_t height = python_system_.GetAttrOfType<long>(*py_mv_image, "height");
            uint32_t num_channels = 3;
            const std::wstring_view mode_str = python_system_.GetAttrOfType<std::wstring_view>(*py_mv_image, "mode");
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

            Texture mv_image(width, height, num_channels);
            std::memcpy(mv_image.Data(), PyBytes_AsString(mask_data.get()), mv_image.DataSize());

            return mv_image;
        }

    private:
        PythonSystem& python_system_;

        PyObjectPtr mv_diffusion_module_;
        PyObjectPtr mv_diffusion_class_;
        PyObjectPtr mv_diffusion_;
        PyObjectPtr mv_diffusion_gen_method_;

        PyObjectPtr pil_module_;
        PyObjectPtr image_class_;
        PyObjectPtr image_frombuffer_method_;
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
