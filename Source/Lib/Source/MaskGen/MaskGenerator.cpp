// Copyright (c) 2024 Minmin Gong
//

#include "MaskGenerator.hpp"

#define Py_BUILD_CORE
#include <Python.h>

#include <iostream>
#include <string_view>

namespace AIHoloImager
{
    class PyObjDeleter
    {
    public:
        void operator()(PyObject* p)
        {
            if (p != nullptr)
            {
                Py_DecRef(p);
            }
        }
    };

    using PyObjectPtr = std::unique_ptr<PyObject, PyObjDeleter>;

    PyObjectPtr MakePyObjectPtr(PyObject* p)
    {
        return PyObjectPtr(p);
    }

    class MaskGenerator::Impl
    {
    public:
        explicit Impl(const std::filesystem::path& exe_dir)
        {
            std::vector<std::wstring> paths;
            paths.push_back(std::filesystem::path(AIHI_PY_STDLIB_DIR).lexically_normal().wstring());
            paths.push_back((std::filesystem::path(AIHI_PY_RUNTIME_LIB_DIR) / "DLLs").lexically_normal().wstring());
            paths.push_back(exe_dir.lexically_normal().wstring());
            paths.push_back((exe_dir / "Python/Lib/site-packages").lexically_normal().wstring());

            PyPreConfig pre_config;
            PyPreConfig_InitIsolatedConfig(&pre_config);

            pre_config.utf8_mode = 1;

            PyStatus status = Py_PreInitialize(&pre_config);
            if (PyStatus_Exception(status))
            {
                Py_ExitStatusException(status);
            }

            PyConfig config;
            PyConfig_InitIsolatedConfig(&config);

            status = PyConfig_SetString(&config, &config.program_name, L"MaskGenerator");
            if (PyStatus_Exception(status))
            {
                PyConfig_Clear(&config);
                Py_ExitStatusException(status);
            }

            config.module_search_paths_set = 1;
            for (const auto& path : paths)
            {
                status = PyWideStringList_Append(&config.module_search_paths, path.c_str());
                if (PyStatus_Exception(status))
                {
                    PyConfig_Clear(&config);
                    Py_ExitStatusException(status);
                }
            }

            status = Py_InitializeFromConfig(&config);
            if (PyStatus_Exception(status))
            {
                PyConfig_Clear(&config);
                Py_ExitStatusException(status);
            }

            PyConfig_Clear(&config);

            mask_generator_module_ = MakePyObjectPtr(PyImport_ImportModule("MaskGenerator"));
            mask_generator_class_ = MakePyObjectPtr(PyObject_GetAttrString(mask_generator_module_.get(), "MaskGenerator"));
            mask_generator_ = MakePyObjectPtr(PyObject_CallObject(mask_generator_class_.get(), nullptr));
            mask_generator_gen_method_ = MakePyObjectPtr(PyObject_GetAttrString(mask_generator_.get(), "Gen"));

            pil_module_ = MakePyObjectPtr(PyImport_ImportModule("PIL"));
            image_class_ = MakePyObjectPtr(PyObject_GetAttrString(pil_module_.get(), "Image"));
            image_frombuffer_method_ = MakePyObjectPtr(PyObject_GetAttrString(image_class_.get(), "frombuffer"));
        }

        ~Impl()
        {
            image_frombuffer_method_.reset();
            image_class_.reset();
            pil_module_.reset();

            mask_generator_gen_method_.reset();
            mask_generator_.reset();
            mask_generator_class_.reset();
            mask_generator_module_.reset();

            Py_Finalize();
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
        PyObjectPtr mask_generator_module_;
        PyObjectPtr mask_generator_class_;
        PyObjectPtr mask_generator_;
        PyObjectPtr mask_generator_gen_method_;

        PyObjectPtr pil_module_;
        PyObjectPtr image_class_;
        PyObjectPtr image_frombuffer_method_;
    };

    MaskGenerator::MaskGenerator(const std::filesystem::path& exe_dir) : impl_(std::make_unique<Impl>(exe_dir))
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
