// Copyright (c) 2025-2026 Minmin Gong
//

#ifdef _DEBUG
    #undef _DEBUG // Stop linking to python<ver>_d.lib
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

#include "SubMConv.hpp"

using namespace AIHoloImager;

PYBIND11_MODULE(AIHoloImagerSubMConv, mod, py::mod_gil_not_used())
{
    pybind11::class_<SubMConv3DHelper>(mod, "SubMConv3DHelper")
        .def(pybind11::init<size_t, torch::Device>())
        .def("BuildCoordsMap", &SubMConv3DHelper::BuildCoordsMap)
        .def("FindAvailableNeighbors", &SubMConv3DHelper::FindAvailableNeighbors);
}
