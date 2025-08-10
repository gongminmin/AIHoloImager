// Copyright (c) 2025 Minmin Gong
//

#pragma once

#include <memory>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4100) // Ignore unreferenced formal parameters
    #pragma warning(disable : 4127) // Ignore constant conditional expression
    #pragma warning(disable : 4244) // Ignore type conversion from `int` to `float`
    #pragma warning(disable : 4251) // Ignore non dll-interface as member
    #pragma warning(disable : 4267) // Ignore type conversion from `size_t` to something else
    #pragma warning(disable : 4324) // Ignore padded structure
    #pragma warning(disable : 4275) // Ignore non dll-interface base class
#endif
#include <torch/types.h>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#include "Gpu/GpuBuffer.hpp"
#include "Gpu/GpuSystem.hpp"
#include "Gpu/GpuTexture.hpp"

namespace AIHoloImager
{
    class TensorConverter
    {
    public:
        TensorConverter(GpuSystem& gpu_system, torch::Device torch_device);
        ~TensorConverter();

        void Convert(
            GpuCommandList& cmd_list, torch::Tensor tensor, GpuBuffer& buff, GpuHeap heap, GpuResourceFlag flags, std::wstring_view name);
        void Convert(GpuCommandList& cmd_list, torch::Tensor tensor, GpuTexture2D& tex, GpuFormat format, GpuResourceFlag flags,
            std::wstring_view name);
        torch::Tensor Convert(GpuCommandList& cmd_list, const GpuBuffer& buff, const torch::IntArrayRef& size, torch::Dtype data_type);
        torch::Tensor Convert(GpuCommandList& cmd_list, const GpuTexture2D& tex);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
