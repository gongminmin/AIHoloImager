// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

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

        PyObjectPtr MakeObject(long value);
        PyObjectPtr MakeObject(std::wstring_view str);
        PyObjectPtr MakeObject(std::span<const std::byte> mem);

        PyObjectPtr MakeTuple(uint32_t size);
        void SetTupleItem(PyObject& tuple, uint32_t index, PyObject& item);
        void SetTupleItem(PyObject& tuple, uint32_t index, PyObjectPtr item);

        template <typename T>
        T Cast(PyObject& object);

        template <typename T>
        T GetAttrOfType(PyObject& module, const char* name)
        {
            return this->Cast<T>(*this->GetAttr(module, name));
        }

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
