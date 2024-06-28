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
    };

    PythonSystem::PythonSystem(const std::filesystem::path& exe_dir) : impl_(std::make_unique<Impl>(exe_dir))
    {
    }

    PythonSystem::~PythonSystem() noexcept = default;

    PythonSystem::PythonSystem(PythonSystem&& other) noexcept = default;
    PythonSystem& PythonSystem::operator=(PythonSystem&& other) noexcept = default;

    PyObjectPtr PythonSystem::Import(const char* name)
    {
        return MakePyObjectPtr(PyImport_ImportModule(name));
    }

    PyObjectPtr PythonSystem::GetAttr(PyObject& module, const char* name)
    {
        return MakePyObjectPtr(PyObject_GetAttrString(&module, name));
    }

    PyObjectPtr PythonSystem::CallObject(PyObject& object)
    {
        return MakePyObjectPtr(PyObject_CallObject(&object, nullptr));
    }

    PyObjectPtr PythonSystem::CallObject(PyObject& object, PyObject& args)
    {
        return MakePyObjectPtr(PyObject_CallObject(&object, &args));
    }

    PyObjectPtr PythonSystem::MakeObject(long value)
    {
        return MakePyObjectPtr(PyLong_FromLong(value));
    }

    PyObjectPtr PythonSystem::MakeObject(std::wstring_view str)
    {
        return MakePyObjectPtr(PyUnicode_FromWideChar(str.data(), str.size()));
    }

    PyObjectPtr PythonSystem::MakeObject(std::span<const std::byte> mem)
    {
        return MakePyObjectPtr(PyBytes_FromStringAndSize(reinterpret_cast<const char*>(mem.data()), mem.size()));
    }

    PyObjectPtr PythonSystem::MakeTuple(uint32_t size)
    {
        return MakePyObjectPtr(PyTuple_New(size));
    }

    void PythonSystem::SetTupleItem(PyObject& tuple, uint32_t index, PyObject& item)
    {
        PyTuple_SetItem(&tuple, index, &item);
    }

    void PythonSystem::SetTupleItem(PyObject& tuple, uint32_t index, PyObjectPtr item)
    {
        PyTuple_SetItem(&tuple, index, item.release());
    }

    template <>
    long PythonSystem::Cast<long>(PyObject& object)
    {
        return PyLong_AsLong(&object);
    }

    template <>
    std::wstring_view PythonSystem::Cast<std::wstring_view>(PyObject& object)
    {
        Py_ssize_t size;
        wchar_t* str = PyUnicode_AsWideCharString(&object, &size);
        return std::wstring_view(str, size);
    }

    template <>
    std::span<const std::byte> PythonSystem::Cast<std::span<const std::byte>>(PyObject& object)
    {
        const Py_ssize_t size = PyBytes_Size(&object);
        const char* data = PyBytes_AsString(&object);
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(data), size);
    }
} // namespace AIHoloImager
