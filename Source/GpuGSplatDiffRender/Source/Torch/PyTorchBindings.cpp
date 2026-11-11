// Copyright (c) 2026 Minmin Gong
//

#ifdef _DEBUG
    #undef _DEBUG // Skip using debug version of refcounts
#endif
#include <Python.h>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4100) // Ignore unreferenced formal parameters
    #pragma warning(disable : 4127) // Ignore constant conditional expression
    #pragma warning(disable : 4244) // Ignore type conversion from `int` to `float`
    #pragma warning(disable : 4251) // Ignore non dll-interface as member
    #pragma warning(disable : 4267) // Ignore type conversion from `size_t` to something else
    #pragma warning(disable : 4275) // Ignore non dll-interface base class
    #pragma warning(disable : 4324) // Ignore padded structure
#endif
#include <torch/csrc/utils/pybind.h>
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

#include "GpuGSplatDiffRenderTorch.hpp"

using namespace AIHoloImager;

PYBIND11_MODULE(AIHoloImagerGpuGSplatDiffRender, mod, py::mod_gil_not_used())
{
    pybind11::class_<GpuGSplatDiffRenderTorch>(mod, "GpuGSplatDiffRenderTorch")
        .def(pybind11::init<size_t, torch::Device>())
        .def("Render", &GpuGSplatDiffRenderTorch::Render, py::arg("positions"), py::arg("scales"), py::arg("rotations"), py::arg("shs"),
            py::arg("opacities"), py::arg("render_settings"));

    pybind11::class_<GpuGSplatDiffRenderTorch::Viewport>(mod, "Viewport")
        .def(pybind11::init<>())
        .def_readwrite("left", &GpuGSplatDiffRenderTorch::Viewport::left)
        .def_readwrite("top", &GpuGSplatDiffRenderTorch::Viewport::top)
        .def_readwrite("width", &GpuGSplatDiffRenderTorch::Viewport::width)
        .def_readwrite("height", &GpuGSplatDiffRenderTorch::Viewport::height);

    pybind11::class_<GpuGSplatDiffRenderTorch::RenderSettings>(mod, "RenderSettings")
        .def(pybind11::init<>())
        .def_readwrite("image_width", &GpuGSplatDiffRenderTorch::RenderSettings::image_width)
        .def_readwrite("image_height", &GpuGSplatDiffRenderTorch::RenderSettings::image_height)
        .def_readwrite("sh_degree", &GpuGSplatDiffRenderTorch::RenderSettings::sh_degree)
        .def_readwrite("model_mtx", &GpuGSplatDiffRenderTorch::RenderSettings::model_mtx)
        .def_readwrite("view_mtx", &GpuGSplatDiffRenderTorch::RenderSettings::view_mtx)
        .def_readwrite("proj_mtx", &GpuGSplatDiffRenderTorch::RenderSettings::proj_mtx)
        .def_readwrite("viewport", &GpuGSplatDiffRenderTorch::RenderSettings::viewport)
        .def_readwrite("kernel_size", &GpuGSplatDiffRenderTorch::RenderSettings::kernel_size)
        .def_readwrite("bg_color", &GpuGSplatDiffRenderTorch::RenderSettings::bg_color);
}
