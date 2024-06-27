// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#define Py_BUILD_CORE
#include <Python.h>

#include "Util/Noncopyable.hpp"

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

    PyObjectPtr MakePyObjectPtr(PyObject* p);

    class PythonSystem
    {
        DISALLOW_COPY_AND_ASSIGN(PythonSystem);

    public:
        explicit PythonSystem(const std::filesystem::path& exe_dir);
        PythonSystem(PythonSystem&& other) noexcept;
        ~PythonSystem() noexcept;

        PythonSystem& operator=(PythonSystem&& other) noexcept;

        PyObjectPtr Import(const char* name);
        PyObjectPtr GetAttr(PyObject& module, const char* name);

        PyObjectPtr CallObject(PyObject& object);
        PyObjectPtr CallObject(PyObject& object, PyObject& args);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
