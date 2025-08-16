// Copyright (c) 2025 Minmin Gong
//

#include "SubMConv.hpp"

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

PYBIND11_MODULE(AIHoloImagerSubMConv, mod)
{
    pybind11::class_<SubMConv3DHelper>(mod, "SubMConv3DHelper")
        .def(pybind11::init<size_t, torch::Device>())
        .def("BuildCoordsMap", &SubMConv3DHelper::BuildCoordsMap)
        .def("FindAvailableNeighbors", &SubMConv3DHelper::FindAvailableNeighbors);
}
