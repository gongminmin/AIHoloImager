// Copyright (c) 2024-2025 Minmin Gong
//

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

#define Py_BUILD_CORE
#include <Python.h>

#include "Base/Noncopyable.hpp"

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
        PythonSystem(bool enable_cuda, const std::filesystem::path& exe_dir);
        PythonSystem(PythonSystem&& other) noexcept;
        ~PythonSystem() noexcept;

        PythonSystem& operator=(PythonSystem&& other) noexcept;

        PyObjectPtr Import(const char* name);
        PyObjectPtr GetAttr(PyObject& module, const char* name);

        PyObjectPtr CallObject(PyObject& object);
        PyObjectPtr CallObject(PyObject& object, PyObject& args);

        PyObjectPtr MakeObject(int32_t value);
        PyObjectPtr MakeObject(uint32_t value);
        PyObjectPtr MakeObject(float value);
        PyObjectPtr MakeObject(std::wstring_view str);
        PyObjectPtr MakeObject(std::span<const std::byte> mem);
        PyObjectPtr MakeObject(void* ptr);

        PyObjectPtr MakeTuple(uint32_t size);
        void SetTupleItem(PyObject& tuple, uint32_t index, PyObject& item);
        void SetTupleItem(PyObject& tuple, uint32_t index, PyObjectPtr item);
        PyObjectPtr GetTupleItem(PyObject& tuple, uint32_t index);

        template <typename T>
        T Cast(PyObject& object);

        std::span<const std::byte> ToBytes(PyObject& object);
        template <typename T>
        std::span<T> ToSpan(PyObject& object)
        {
            const auto bytes = ToBytes(object);
            return std::span<T>(reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T));
        }

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
