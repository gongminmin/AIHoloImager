// Copyright (c) 2024 Minmin Gong
//

#include "MaskGenerator.hpp"

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
                auto args = MakePyObjectPtr(PyTuple_New(3));
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

                    PyTuple_SetItem(args.get(), 0, PyUnicode_FromWideChar(mode.data(), mode.size()));
                }
                {
                    auto size = MakePyObjectPtr(PyTuple_New(2));
                    {
                        PyTuple_SetItem(size.get(), 0, PyLong_FromLong(input_image.Width()));
                        PyTuple_SetItem(size.get(), 1, PyLong_FromLong(input_image.Height()));
                    }
                    PyTuple_SetItem(args.get(), 1, size.release());
                }
                {
                    PyTuple_SetItem(args.get(), 2,
                        PyBytes_FromStringAndSize(reinterpret_cast<const char*>(input_image.Data()), input_image.DataSize()));
                }

                py_input_image = MakePyObjectPtr(PyObject_CallObject(image_frombuffer_method_.get(), args.get()));
            }

            auto args = MakePyObjectPtr(PyTuple_New(1));
            {
                PyTuple_SetItem(args.get(), 0, py_input_image.get());
            }

            auto py_mask_image = MakePyObjectPtr(PyObject_CallObject(mask_generator_gen_method_.get(), args.get()));
            auto tobytes_method = MakePyObjectPtr(PyObject_GetAttrString(py_mask_image.get(), "tobytes"));
            auto mask_data = MakePyObjectPtr(PyObject_CallObject(tobytes_method.get(), nullptr));

            const uint32_t width = PyLong_AsLong(PyObject_GetAttrString(py_mask_image.get(), "width"));
            const uint32_t height = PyLong_AsLong(PyObject_GetAttrString(py_mask_image.get(), "height"));
            uint32_t num_channels = 3;
            std::wstring_view mode_str = PyUnicode_AsWideCharString(PyObject_GetAttrString(py_mask_image.get(), "mode"), nullptr);
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
            std::memcpy(mask_image.Data(), PyBytes_AsString(mask_data.get()), mask_image.DataSize());

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
