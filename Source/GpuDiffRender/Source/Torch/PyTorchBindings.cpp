// Copyright (c) 2025 Minmin Gong
//

#include "GpuDiffRenderTorch.hpp"

#ifdef _DEBUG
    #undef _DEBUG // Stop linking to python<ver>_d.lib
#endif
#include <Python.h>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4251) // Ignore non dll-interface as member
#endif
#include <torch/csrc/utils/pybind.h>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

using namespace AIHoloImager;

PYBIND11_MODULE(AIHoloImagerGpuDiffRender, mod)
{
    pybind11::class_<GpuDiffRenderTorch>(mod, "GpuDiffRenderTorch")
        .def(pybind11::init<size_t, torch::Device>())
        .def("Rasterize", &GpuDiffRenderTorch::Rasterize)
        .def("Interpolate", &GpuDiffRenderTorch::Interpolate)
        .def("AntiAliasConstructOppositeVertices", &GpuDiffRenderTorch::AntiAliasConstructOppositeVertices)
        .def("AntiAlias", &GpuDiffRenderTorch::AntiAlias);

    pybind11::class_<GpuDiffRenderTorch::AntiAliasOppositeVertices>(mod, "AntiAliasOppositeVertices");
}
