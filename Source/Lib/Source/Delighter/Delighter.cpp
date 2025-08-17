// Copyright (c) 2025 Minmin Gong
//

#include "Delighter.hpp"

#include <future>

#include "Util/PerfProfiler.hpp"

namespace AIHoloImager
{
    class Delighter::Impl
    {
    public:
        explicit Impl(AIHoloImagerInternal& aihi) : aihi_(aihi)
        {
            py_init_future_ = std::async(std::launch::async, [this] {
                PerfRegion init_async_perf(aihi_.PerfProfilerInstance(), "Delighter init (async)");

                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();

                delighter_module_ = python_system.Import("Delighter");
                delighter_class_ = python_system.GetAttr(*delighter_module_, "Delighter");
                delighter_ = python_system.CallObject(*delighter_class_);
                delighter_process_method_ = python_system.GetAttr(*delighter_, "Process");
            });
        }

        ~Impl()
        {
            PerfRegion destroy_perf(aihi_.PerfProfilerInstance(), "Delighter destroy");

            py_init_future_.wait();

            PythonSystem::GilGuard guard;

            auto& python_system = aihi_.PythonSystemInstance();
            auto delighter_destroy_method = python_system.GetAttr(*delighter_, "Destroy");
            python_system.CallObject(*delighter_destroy_method);

            delighter_destroy_method.reset();
            delighter_process_method_.reset();
            delighter_.reset();
            delighter_class_.reset();
            delighter_module_.reset();
        }

        Texture Process(const Texture& image, const glm::uvec4& roi, glm::uvec2& offset)
        {
            PerfRegion process_perf(aihi_.PerfProfilerInstance(), "Delighter process");

            const uint32_t width = image.Width();
            const uint32_t height = image.Height();

            constexpr uint32_t Gap = 32;
            glm::uvec4 expanded_roi;
            expanded_roi.x = std::max(static_cast<uint32_t>(std::floor(roi.x)) - Gap, 0U);
            expanded_roi.y = std::max(static_cast<uint32_t>(std::floor(roi.y)) - Gap, 0U);
            expanded_roi.z = std::min(static_cast<uint32_t>(std::ceil(roi.z)) + Gap, width);
            expanded_roi.w = std::min(static_cast<uint32_t>(std::ceil(roi.w)) + Gap, height);

            offset = glm::uvec2(expanded_roi.x, expanded_roi.y);

            const uint32_t roi_width = expanded_roi.z - expanded_roi.x;
            const uint32_t roi_height = expanded_roi.w - expanded_roi.y;
            const uint32_t fmt_size = FormatSize(image.Format());
            Texture roi_image(roi_width, roi_height, image.Format());
            {
                std::byte* dst = roi_image.Data();
                const std::byte* src = &image.Data()[(expanded_roi.y * width + expanded_roi.x) * fmt_size];
                const uint32_t dst_row_pitch = roi_width * fmt_size;
                const uint32_t src_row_pitch = width * fmt_size;
                for (uint32_t y = 0; y < roi_height; ++y)
                {
                    std::memcpy(dst, src, dst_row_pitch);
                    dst += dst_row_pitch;
                    src += src_row_pitch;
                }
            }

            {
                PerfRegion wait_perf(aihi_.PerfProfilerInstance(), "Wait for init");
                py_init_future_.wait();
            }

            Texture out_image;
            {
                PythonSystem::GilGuard guard;

                auto& python_system = aihi_.PythonSystemInstance();
                auto args = python_system.MakeTuple(4);
                {
                    auto py_image = python_system.MakeObject(
                        std::span<const std::byte>(reinterpret_cast<const std::byte*>(roi_image.Data()), roi_image.DataSize()));
                    python_system.SetTupleItem(*args, 0, std::move(py_image));

                    python_system.SetTupleItem(*args, 1, python_system.MakeObject(roi_width));
                    python_system.SetTupleItem(*args, 2, python_system.MakeObject(roi_height));
                    python_system.SetTupleItem(*args, 3, python_system.MakeObject(FormatChannels(roi_image.Format())));
                }

                const auto output_roi_image = python_system.CallObject(*delighter_process_method_, *args);

                out_image = Texture(roi_width, roi_height, ElementFormat::RGBA8_UNorm);
                std::byte* dst = reinterpret_cast<std::byte*>(out_image.Data());
                const std::byte* src = python_system.ToBytes(*output_roi_image).data();
                const std::byte* src_mask = &image.Data()[(expanded_roi.y * width + expanded_roi.x) * fmt_size];
                for (uint32_t y = 0; y < roi_height; ++y)
                {
                    for (uint32_t x = 0; x < roi_width; ++x)
                    {
                        const uint32_t img_offset = y * roi_width + x;
                        dst[img_offset * 4 + 0] = src[img_offset * 3 + 0];
                        dst[img_offset * 4 + 1] = src[img_offset * 3 + 1];
                        dst[img_offset * 4 + 2] = src[img_offset * 3 + 2];
                        dst[img_offset * 4 + 3] = src_mask[(y * width + x) * fmt_size + 3];
                    }
                }
            }

            return out_image;
        }

    private:
        AIHoloImagerInternal& aihi_;

        PyObjectPtr delighter_module_;
        PyObjectPtr delighter_class_;
        PyObjectPtr delighter_;
        PyObjectPtr delighter_process_method_;
        std::future<void> py_init_future_;
    };

    Delighter::Delighter(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    Delighter::~Delighter() noexcept = default;

    Delighter::Delighter(Delighter&& other) noexcept = default;
    Delighter& Delighter::operator=(Delighter&& other) noexcept = default;

    Texture Delighter::Process(const Texture& image, const glm::uvec4& roi, glm::uvec2& offset)
    {
        return impl_->Process(image, roi, offset);
    }
} // namespace AIHoloImager
