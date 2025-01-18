// Copyright (c) 2025 Minmin Gong
//

#include "Delighter.hpp"

namespace AIHoloImager
{
    class Delighter::Impl
    {
    public:
        Impl(PythonSystem& python_system) : python_system_(python_system)
        {
            delighter_module_ = python_system_.Import("Delighter");
            delighter_class_ = python_system_.GetAttr(*delighter_module_, "Delighter");
            delighter_ = python_system_.CallObject(*delighter_class_);
            delighter_process_method_ = python_system_.GetAttr(*delighter_, "Process");
        }

        ~Impl()
        {
            auto delighter_destroy_method = python_system_.GetAttr(*delighter_, "Destroy");
            python_system_.CallObject(*delighter_destroy_method);
        }

        void ProcessInPlace(Texture& inout_image, const glm::uvec4& roi)
        {
            const uint32_t width = inout_image.Width();
            const uint32_t height = inout_image.Height();

            constexpr uint32_t Gap = 32;
            glm::uvec4 expanded_roi;
            expanded_roi.x = std::max(static_cast<uint32_t>(std::floor(roi.x)) - Gap, 0U);
            expanded_roi.y = std::max(static_cast<uint32_t>(std::floor(roi.y)) - Gap, 0U);
            expanded_roi.z = std::min(static_cast<uint32_t>(std::ceil(roi.z)) + Gap, width);
            expanded_roi.w = std::min(static_cast<uint32_t>(std::ceil(roi.w)) + Gap, height);

            const uint32_t roi_width = expanded_roi.z - expanded_roi.x;
            const uint32_t roi_height = expanded_roi.w - expanded_roi.y;
            const uint32_t fmt_size = FormatSize(inout_image.Format());
            Texture roi_image(roi_width, roi_height, inout_image.Format());
            {
                std::byte* dst = roi_image.Data();
                const std::byte* src = &inout_image.Data()[(expanded_roi.y * inout_image.Width() + expanded_roi.x) * fmt_size];
                const uint32_t dst_row_pitch = roi_width * fmt_size;
                const uint32_t src_row_pitch = inout_image.Width() * fmt_size;
                for (uint32_t y = 0; y < roi_height; ++y)
                {
                    std::memcpy(dst, src, dst_row_pitch);
                    dst += dst_row_pitch;
                    src += src_row_pitch;
                }
            }

            auto args = python_system_.MakeTuple(4);
            {
                auto image = python_system_.MakeObject(
                    std::span<const std::byte>(reinterpret_cast<const std::byte*>(roi_image.Data()), roi_image.DataSize()));
                python_system_.SetTupleItem(*args, 0, std::move(image));

                python_system_.SetTupleItem(*args, 1, python_system_.MakeObject(roi_width));
                python_system_.SetTupleItem(*args, 2, python_system_.MakeObject(roi_height));
                python_system_.SetTupleItem(*args, 3, python_system_.MakeObject(FormatChannels(roi_image.Format())));
            }

            const auto output_roi_image = python_system_.CallObject(*delighter_process_method_, *args);
            {
                std::byte* dst = &inout_image.Data()[(expanded_roi.y * inout_image.Width() + expanded_roi.x) * fmt_size];
                const std::byte* src = python_system_.ToBytes(*output_roi_image).data();
                const uint32_t dst_row_pitch = inout_image.Width() * fmt_size;
                const uint32_t src_row_pitch = roi_width * fmt_size;
                for (uint32_t y = 0; y < roi_height; ++y)
                {
                    std::memcpy(dst, src, src_row_pitch);
                    dst += dst_row_pitch;
                    src += src_row_pitch;
                }
            }
        }

    private:
        PythonSystem& python_system_;

        PyObjectPtr delighter_module_;
        PyObjectPtr delighter_class_;
        PyObjectPtr delighter_;
        PyObjectPtr delighter_process_method_;
    };

    Delighter::Delighter(PythonSystem& python_system) : impl_(std::make_unique<Impl>(python_system))
    {
    }

    Delighter::~Delighter() noexcept = default;

    Delighter::Delighter(Delighter&& other) noexcept = default;
    Delighter& Delighter::operator=(Delighter&& other) noexcept = default;

    void Delighter::ProcessInPlace(Texture& inout_image, const glm::uvec4& roi)
    {
        impl_->ProcessInPlace(inout_image, roi);
    }
} // namespace AIHoloImager
