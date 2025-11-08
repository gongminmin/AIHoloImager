// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

#include "Base/MiniWindows.hpp"
#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"

#ifdef _WIN32
    #define AIHI_SYMBOL_EXPORT __declspec(dllexport)
    #define AIHI_SYMBOL_IMPORT __declspec(dllimport)
#else
    #define AIHI_SYMBOL_EXPORT __attribute__((visibility("default")))
    #define AIHI_SYMBOL_IMPORT
#endif

#ifdef AIHoloImagerTensorConverter_EXPORTS
    #define AIHI_TC_API AIHI_SYMBOL_EXPORT
#else
    #define AIHI_TC_API AIHI_SYMBOL_IMPORT
#endif

typedef struct _object PyObject;

namespace c10
{
    struct Device;
    template <typename T>
    class ArrayRef;
    using IntArrayRef = c10::ArrayRef<int64_t>;
    enum class ScalarType : int8_t;
} // namespace c10

namespace at
{
    class Tensor;
}

namespace torch
{
    using Device = c10::Device;
    using Tensor = at::Tensor;
    using IntArrayRef = c10::IntArrayRef;
    using Dtype = c10::ScalarType;
} // namespace torch

namespace AIHoloImager
{
    class TensorConverter
    {
    public:
        AIHI_TC_API TensorConverter(GpuSystem& gpu_system, std::string_view torch_device);
        AIHI_TC_API TensorConverter(GpuSystem& gpu_system, const torch::Device& torch_device);
        AIHI_TC_API ~TensorConverter() noexcept;

        AIHI_TC_API void Convert(GpuCommandList& cmd_list, const torch::Tensor& tensor, GpuBuffer& buff, GpuHeap heap,
            GpuResourceFlag flags, std::string_view name) const;
        AIHI_TC_API void Convert(GpuCommandList& cmd_list, const torch::Tensor& tensor, GpuTexture2D& tex, GpuFormat format,
            GpuResourceFlag flags, std::string_view name) const;
        AIHI_TC_API torch::Tensor Convert(
            GpuCommandList& cmd_list, const GpuBuffer& buff, const torch::IntArrayRef& size, torch::Dtype data_type) const;
        AIHI_TC_API torch::Tensor Convert(GpuCommandList& cmd_list, const GpuTexture2D& tex) const;

        AIHI_TC_API void ConvertPy(GpuCommandList& cmd_list, const PyObject& py_tensor, GpuBuffer& buff, GpuHeap heap,
            GpuResourceFlag flags, std::string_view name) const;
        AIHI_TC_API void ConvertPy(GpuCommandList& cmd_list, const PyObject& py_tensor, GpuTexture2D& tex, GpuFormat format,
            GpuResourceFlag flags, std::string_view name) const;
        AIHI_TC_API PyObject* ConvertPy(
            GpuCommandList& cmd_list, const GpuBuffer& buff, const torch::IntArrayRef& size, torch::Dtype data_type) const;
        AIHI_TC_API PyObject* ConvertPy(GpuCommandList& cmd_list, const GpuTexture2D& tex) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
