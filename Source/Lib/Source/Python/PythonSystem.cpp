// Copyright (c) 2024 Minmin Gong
//

#include "PythonSystem.hpp"

namespace AIHoloImager
{
    PyObjectPtr MakePyObjectPtr(PyObject* p)
    {
        return PyObjectPtr(p);
    }

    class PythonSystem::Impl
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

            status = PyConfig_SetString(&config, &config.program_name, L"AIHoloImager");
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
        }

        ~Impl()
        {
            Py_Finalize();
        }

        PyObjectPtr Import(const char* name)
        {
            return MakePyObjectPtr(PyImport_ImportModule(name));
        }

        PyObjectPtr GetAttr(PyObject& module, const char* name)
        {
            return MakePyObjectPtr(PyObject_GetAttrString(&module, name));
        }

        PyObjectPtr CallObject(PyObject& object)
        {
            return MakePyObjectPtr(PyObject_CallObject(&object, nullptr));
        }

        PyObjectPtr CallObject(PyObject& object, PyObject& args)
        {
            return MakePyObjectPtr(PyObject_CallObject(&object, &args));
        }
    };

    PythonSystem::PythonSystem(const std::filesystem::path& exe_dir) : impl_(std::make_unique<Impl>(exe_dir))
    {
    }

    PythonSystem::~PythonSystem() noexcept = default;

    PythonSystem::PythonSystem(PythonSystem&& other) noexcept = default;
    PythonSystem& PythonSystem::operator=(PythonSystem&& other) noexcept = default;

    PyObjectPtr PythonSystem::Import(const char* name)
    {
        return impl_->Import(name);
    }

    PyObjectPtr PythonSystem::GetAttr(PyObject& module, const char* name)
    {
        return impl_->GetAttr(module, name);
    }

    PyObjectPtr PythonSystem::CallObject(PyObject& object)
    {
        return impl_->CallObject(object);
    }

    PyObjectPtr PythonSystem::CallObject(PyObject& object, PyObject& args)
    {
        return impl_->CallObject(object, args);
    }
} // namespace AIHoloImager
