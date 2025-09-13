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
        .def("Rasterize", &GpuDiffRenderTorch::Rasterize, py::arg("positions"), py::arg("indices"), py::arg("resolution"),
            py::arg("viewport") = py::none())
        .def("Interpolate", &GpuDiffRenderTorch::Interpolate, py::arg("vtx_attribs"), py::arg("barycentric"), py::arg("prim_id"),
            py::arg("indices"))
        .def("AntiAliasConstructOppositeVertices", &GpuDiffRenderTorch::AntiAliasConstructOppositeVertices, py::arg("indices"))
        .def("AntiAlias", &GpuDiffRenderTorch::AntiAlias, py::arg("shading"), py::arg("prim_id"), py::arg("positions"), py::arg("indices"),
            py::arg("viewport") = py::none(), py::arg("opposite_vertices") = py::none())
        .def("Texture", &GpuDiffRenderTorch::Texture, py::arg("texture"), py::arg("prim_id"), py::arg("uv"), py::arg("filter"),
            py::arg("address_mode"));

    pybind11::class_<GpuDiffRenderTorch::Viewport>(mod, "Viewport")
        .def(pybind11::init<>())
        .def_readwrite("left", &GpuDiffRenderTorch::Viewport::left)
        .def_readwrite("top", &GpuDiffRenderTorch::Viewport::top)
        .def_readwrite("width", &GpuDiffRenderTorch::Viewport::width)
        .def_readwrite("height", &GpuDiffRenderTorch::Viewport::height);

    pybind11::class_<GpuDiffRenderTorch::AntiAliasOppositeVertices>(mod, "AntiAliasOppositeVertices").def(pybind11::init<>());
}
