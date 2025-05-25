// Copyright (c) 2024-2025 Minmin Gong
//

#include "MarchingCubes.hpp"

#include <memory>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "Gpu/GpuBufferHelper.hpp"
#include "Gpu/GpuCommandList.hpp"
#include "Gpu/GpuResourceViews.hpp"
#include "Gpu/GpuShader.hpp"

#include "CompiledShader/MeshGen/MarchingCubes/CalcCubeIndicesCs.h"
#include "CompiledShader/MeshGen/MarchingCubes/GenVerticesIndicesCs.h"
#include "CompiledShader/MeshGen/MarchingCubes/ProcessNonEmptyCubesCs.h"

namespace
{
    constexpr uint16_t EdgeTable[] = {0x000, 0x109, 0x203, 0x30A, 0x406, 0x50F, 0x605, 0x70C, 0x80C, 0x905, 0xA0F, 0xB06, 0xC0A, 0xD03,
        0xE09, 0xF00, 0x190, 0x099, 0x393, 0x29A, 0x596, 0x49F, 0x795, 0x69C, 0x99C, 0x895, 0xB9F, 0xA96, 0xD9A, 0xC93, 0xF99, 0xE90, 0x230,
        0x339, 0x033, 0x13A, 0x636, 0x73F, 0x435, 0x53C, 0xA3C, 0xB35, 0x83F, 0x936, 0xE3A, 0xF33, 0xC39, 0xD30, 0x3A0, 0x2A9, 0x1A3, 0x0AA,
        0x7A6, 0x6AF, 0x5A5, 0x4AC, 0xBAC, 0xAA5, 0x9AF, 0x8A6, 0xFAA, 0xEA3, 0xDA9, 0xCA0, 0x460, 0x569, 0x663, 0x76A, 0x066, 0x16F, 0x265,
        0x36C, 0xC6C, 0xD65, 0xE6F, 0xF66, 0x86A, 0x963, 0xA69, 0xB60, 0x5F0, 0x4F9, 0x7F3, 0x6FA, 0x1F6, 0x0FF, 0x3F5, 0x2FC, 0xDFC, 0xCF5,
        0xFFF, 0xEF6, 0x9FA, 0x8F3, 0xBF9, 0xAF0, 0x650, 0x759, 0x453, 0x55A, 0x256, 0x35F, 0x055, 0x15C, 0xE5C, 0xF55, 0xC5F, 0xD56, 0xA5A,
        0xB53, 0x859, 0x950, 0x7C0, 0x6C9, 0x5C3, 0x4CA, 0x3C6, 0x2CF, 0x1C5, 0x0CC, 0xFCC, 0xEC5, 0xDCF, 0xCC6, 0xBCA, 0xAC3, 0x9C9, 0x8C0,
        0x8C0, 0x9C9, 0xAC3, 0xBCA, 0xCC6, 0xDCF, 0xEC5, 0xFCC, 0x0CC, 0x1C5, 0x2CF, 0x3C6, 0x4CA, 0x5C3, 0x6C9, 0x7C0, 0x950, 0x859, 0xB53,
        0xA5A, 0xD56, 0xC5F, 0xF55, 0xE5C, 0x15C, 0x055, 0x35F, 0x256, 0x55A, 0x453, 0x759, 0x650, 0xAF0, 0xBF9, 0x8F3, 0x9FA, 0xEF6, 0xFFF,
        0xCF5, 0xDFC, 0x2FC, 0x3F5, 0x0FF, 0x1F6, 0x6FA, 0x7F3, 0x4F9, 0x5F0, 0xB60, 0xA69, 0x963, 0x86A, 0xF66, 0xE6F, 0xD65, 0xC6C, 0x36C,
        0x265, 0x16F, 0x066, 0x76A, 0x663, 0x569, 0x460, 0xCA0, 0xDA9, 0xEA3, 0xFAA, 0x8A6, 0x9AF, 0xAA5, 0xBAC, 0x4AC, 0x5A5, 0x6AF, 0x7A6,
        0x0AA, 0x1A3, 0x2A9, 0x3A0, 0xD30, 0xC39, 0xF33, 0xE3A, 0x936, 0x83F, 0xB35, 0xA3C, 0x53C, 0x435, 0x73F, 0x636, 0x13A, 0x033, 0x339,
        0x230, 0xE90, 0xF99, 0xC93, 0xD9A, 0xA96, 0xB9F, 0x895, 0x99C, 0x69C, 0x795, 0x49F, 0x596, 0x29A, 0x393, 0x099, 0x190, 0xF00, 0xE09,
        0xD03, 0xC0A, 0xB06, 0xA0F, 0x905, 0x80C, 0x70C, 0x605, 0x50F, 0x406, 0x30A, 0x203, 0x109, 0x000};
    static_assert(std::size(EdgeTable) == 256);

    constexpr uint8_t TriangleTable[][16] = {
        {0},
        {3, 0, 3, 8},
        {3, 0, 9, 1},
        {6, 1, 3, 8, 9, 1, 8},
        {3, 1, 10, 2},
        {6, 0, 3, 8, 1, 10, 2},
        {6, 9, 10, 2, 0, 9, 2},
        {9, 2, 3, 8, 2, 8, 10, 10, 8, 9},
        {3, 3, 2, 11},
        {6, 0, 2, 11, 8, 0, 11},
        {6, 1, 0, 9, 2, 11, 3},
        {9, 1, 2, 11, 1, 11, 9, 9, 11, 8},
        {6, 3, 1, 10, 11, 3, 10},
        {9, 0, 1, 10, 0, 10, 8, 8, 10, 11},
        {9, 3, 0, 9, 3, 9, 11, 11, 9, 10},
        {6, 9, 10, 8, 10, 11, 8},
        {3, 4, 8, 7},
        {6, 4, 0, 3, 7, 4, 3},
        {6, 0, 9, 1, 8, 7, 4},
        {9, 4, 9, 1, 4, 1, 7, 7, 1, 3},
        {6, 1, 10, 2, 8, 7, 4},
        {9, 3, 7, 4, 3, 4, 0, 1, 10, 2},
        {9, 9, 10, 2, 9, 2, 0, 8, 7, 4},
        {12, 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9},
        {6, 8, 7, 4, 3, 2, 11},
        {9, 11, 7, 4, 11, 4, 2, 2, 4, 0},
        {9, 9, 1, 0, 8, 7, 4, 2, 11, 3},
        {12, 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2},
        {9, 3, 1, 10, 3, 10, 11, 7, 4, 8},
        {12, 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11},
        {12, 4, 8, 7, 9, 11, 0, 9, 10, 11, 11, 3, 0},
        {9, 4, 11, 7, 4, 9, 11, 9, 10, 11},
        {3, 9, 4, 5},
        {6, 9, 4, 5, 0, 3, 8},
        {6, 0, 4, 5, 1, 0, 5},
        {9, 8, 4, 5, 8, 5, 3, 3, 5, 1},
        {6, 1, 10, 2, 9, 4, 5},
        {9, 3, 8, 0, 1, 10, 2, 4, 5, 9},
        {9, 5, 10, 2, 5, 2, 4, 4, 2, 0},
        {12, 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4},
        {6, 9, 4, 5, 2, 11, 3},
        {9, 0, 2, 11, 0, 11, 8, 4, 5, 9},
        {9, 0, 4, 5, 0, 5, 1, 2, 11, 3},
        {12, 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8},
        {9, 10, 11, 3, 10, 3, 1, 9, 4, 5},
        {12, 4, 5, 9, 0, 1, 8, 8, 1, 10, 8, 10, 11},
        {12, 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0},
        {9, 5, 8, 4, 5, 10, 8, 10, 11, 8},
        {6, 9, 8, 7, 5, 9, 7},
        {9, 9, 0, 3, 9, 3, 5, 5, 3, 7},
        {9, 0, 8, 7, 0, 7, 1, 1, 7, 5},
        {6, 1, 3, 5, 3, 7, 5},
        {9, 9, 8, 7, 9, 7, 5, 10, 2, 1},
        {12, 10, 2, 1, 9, 0, 5, 5, 0, 3, 5, 3, 7},
        {12, 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5},
        {9, 2, 5, 10, 2, 3, 5, 3, 7, 5},
        {9, 7, 5, 9, 7, 9, 8, 3, 2, 11},
        {12, 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7},
        {12, 2, 11, 3, 0, 8, 1, 1, 8, 7, 1, 7, 5},
        {9, 11, 1, 2, 11, 7, 1, 7, 5, 1},
        {12, 9, 8, 5, 8, 7, 5, 10, 3, 1, 10, 11, 3},
        {15, 5, 0, 7, 5, 9, 0, 7, 0, 11, 1, 10, 0, 11, 0, 10},
        {15, 11, 0, 10, 11, 3, 0, 10, 0, 5, 8, 7, 0, 5, 0, 7},
        {6, 11, 5, 10, 7, 5, 11},
        {3, 10, 5, 6},
        {6, 0, 3, 8, 5, 6, 10},
        {6, 9, 1, 0, 5, 6, 10},
        {9, 1, 3, 8, 1, 8, 9, 5, 6, 10},
        {6, 1, 5, 6, 2, 1, 6},
        {9, 1, 5, 6, 1, 6, 2, 3, 8, 0},
        {9, 9, 5, 6, 9, 6, 0, 0, 6, 2},
        {12, 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2},
        {6, 2, 11, 3, 10, 5, 6},
        {9, 11, 8, 0, 11, 0, 2, 10, 5, 6},
        {9, 0, 9, 1, 2, 11, 3, 5, 6, 10},
        {12, 5, 6, 10, 1, 2, 9, 9, 2, 11, 9, 11, 8},
        {9, 6, 11, 3, 6, 3, 5, 5, 3, 1},
        {12, 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11},
        {12, 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5},
        {9, 6, 9, 5, 6, 11, 9, 11, 8, 9},
        {6, 5, 6, 10, 4, 8, 7},
        {9, 4, 0, 3, 4, 3, 7, 6, 10, 5},
        {9, 1, 0, 9, 5, 6, 10, 8, 7, 4},
        {12, 10, 5, 6, 1, 7, 9, 1, 3, 7, 7, 4, 9},
        {9, 6, 2, 1, 6, 1, 5, 4, 8, 7},
        {12, 1, 5, 2, 5, 6, 2, 3, 4, 0, 3, 7, 4},
        {12, 8, 7, 4, 9, 5, 0, 0, 5, 6, 0, 6, 2},
        {15, 7, 9, 3, 7, 4, 9, 3, 9, 2, 5, 6, 9, 2, 9, 6},
        {9, 3, 2, 11, 7, 4, 8, 10, 5, 6},
        {12, 5, 6, 10, 4, 2, 7, 4, 0, 2, 2, 11, 7},
        {12, 0, 9, 1, 4, 8, 7, 2, 11, 3, 5, 6, 10},
        {15, 9, 1, 2, 9, 2, 11, 9, 11, 4, 7, 4, 11, 5, 6, 10},
        {12, 8, 7, 4, 3, 5, 11, 3, 1, 5, 5, 6, 11},
        {15, 5, 11, 1, 5, 6, 11, 1, 11, 0, 7, 4, 11, 0, 11, 4},
        {15, 0, 9, 5, 0, 5, 6, 0, 6, 3, 11, 3, 6, 8, 7, 4},
        {12, 6, 9, 5, 6, 11, 9, 4, 9, 7, 7, 9, 11},
        {6, 10, 9, 4, 6, 10, 4},
        {9, 4, 6, 10, 4, 10, 9, 0, 3, 8},
        {9, 10, 1, 0, 10, 0, 6, 6, 0, 4},
        {12, 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1},
        {9, 1, 9, 4, 1, 4, 2, 2, 4, 6},
        {12, 3, 8, 0, 1, 9, 2, 2, 9, 4, 2, 4, 6},
        {6, 0, 4, 2, 4, 6, 2},
        {9, 8, 2, 3, 8, 4, 2, 4, 6, 2},
        {9, 10, 9, 4, 10, 4, 6, 11, 3, 2},
        {12, 0, 2, 8, 2, 11, 8, 4, 10, 9, 4, 6, 10},
        {12, 3, 2, 11, 0, 6, 1, 0, 4, 6, 6, 10, 1},
        {15, 6, 1, 4, 6, 10, 1, 4, 1, 8, 2, 11, 1, 8, 1, 11},
        {12, 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6},
        {15, 8, 1, 11, 8, 0, 1, 11, 1, 6, 9, 4, 1, 6, 1, 4},
        {9, 3, 6, 11, 3, 0, 6, 0, 4, 6},
        {6, 6, 8, 4, 11, 8, 6},
        {9, 7, 6, 10, 7, 10, 8, 8, 10, 9},
        {12, 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7},
        {12, 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8},
        {9, 10, 7, 6, 10, 1, 7, 1, 3, 7},
        {12, 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6},
        {15, 2, 9, 6, 2, 1, 9, 6, 9, 7, 0, 3, 9, 7, 9, 3},
        {9, 7, 0, 8, 7, 6, 0, 6, 2, 0},
        {6, 7, 2, 3, 6, 2, 7},
        {12, 2, 11, 3, 10, 8, 6, 10, 9, 8, 8, 7, 6},
        {15, 2, 7, 0, 2, 11, 7, 0, 7, 9, 6, 10, 7, 9, 7, 10},
        {15, 1, 0, 8, 1, 8, 7, 1, 7, 10, 6, 10, 7, 2, 11, 3},
        {12, 11, 1, 2, 11, 7, 1, 10, 1, 6, 6, 1, 7},
        {15, 8, 6, 9, 8, 7, 6, 9, 6, 1, 11, 3, 6, 1, 6, 3},
        {6, 0, 1, 9, 11, 7, 6},
        {12, 7, 0, 8, 7, 6, 0, 3, 0, 11, 11, 0, 6},
        {3, 7, 6, 11},
        {3, 7, 11, 6},
        {6, 3, 8, 0, 11, 6, 7},
        {6, 0, 9, 1, 11, 6, 7},
        {9, 8, 9, 1, 8, 1, 3, 11, 6, 7},
        {6, 10, 2, 1, 6, 7, 11},
        {9, 1, 10, 2, 3, 8, 0, 6, 7, 11},
        {9, 2, 0, 9, 2, 9, 10, 6, 7, 11},
        {12, 6, 7, 11, 2, 3, 10, 10, 3, 8, 10, 8, 9},
        {6, 7, 3, 2, 6, 7, 2},
        {9, 7, 8, 0, 7, 0, 6, 6, 0, 2},
        {9, 2, 6, 7, 2, 7, 3, 0, 9, 1},
        {12, 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7},
        {9, 10, 6, 7, 10, 7, 1, 1, 7, 3},
        {12, 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0},
        {12, 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10},
        {9, 7, 10, 6, 7, 8, 10, 8, 9, 10},
        {6, 6, 4, 8, 11, 6, 8},
        {9, 3, 11, 6, 3, 6, 0, 0, 6, 4},
        {9, 8, 11, 6, 8, 6, 4, 9, 1, 0},
        {12, 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3},
        {9, 6, 4, 8, 6, 8, 11, 2, 1, 10},
        {12, 1, 10, 2, 3, 11, 0, 0, 11, 6, 0, 6, 4},
        {12, 4, 8, 11, 4, 11, 6, 0, 9, 2, 2, 9, 10},
        {15, 10, 3, 9, 10, 2, 3, 9, 3, 4, 11, 6, 3, 4, 3, 6},
        {9, 8, 3, 2, 8, 2, 4, 4, 2, 6},
        {6, 0, 2, 4, 4, 2, 6},
        {12, 1, 0, 9, 2, 4, 3, 2, 6, 4, 4, 8, 3},
        {9, 1, 4, 9, 1, 2, 4, 2, 6, 4},
        {12, 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10},
        {9, 10, 0, 1, 10, 6, 0, 6, 4, 0},
        {15, 4, 3, 6, 4, 8, 3, 6, 3, 10, 0, 9, 3, 10, 3, 9},
        {6, 10, 4, 9, 6, 4, 10},
        {6, 4, 5, 9, 7, 11, 6},
        {9, 0, 3, 8, 4, 5, 9, 11, 6, 7},
        {9, 5, 1, 0, 5, 0, 4, 7, 11, 6},
        {12, 11, 6, 7, 8, 4, 3, 3, 4, 5, 3, 5, 1},
        {9, 9, 4, 5, 10, 2, 1, 7, 11, 6},
        {12, 6, 7, 11, 1, 10, 2, 0, 3, 8, 4, 5, 9},
        {12, 7, 11, 6, 5, 10, 4, 4, 10, 2, 4, 2, 0},
        {15, 3, 8, 4, 3, 4, 5, 3, 5, 2, 10, 2, 5, 11, 6, 7},
        {9, 7, 3, 2, 7, 2, 6, 5, 9, 4},
        {12, 9, 4, 5, 0, 6, 8, 0, 2, 6, 6, 7, 8},
        {12, 3, 2, 6, 3, 6, 7, 1, 0, 5, 5, 0, 4},
        {15, 6, 8, 2, 6, 7, 8, 2, 8, 1, 4, 5, 8, 1, 8, 5},
        {12, 9, 4, 5, 10, 6, 1, 1, 6, 7, 1, 7, 3},
        {15, 1, 10, 6, 1, 6, 7, 1, 7, 0, 8, 0, 7, 9, 4, 5},
        {15, 4, 10, 0, 4, 5, 10, 0, 10, 3, 6, 7, 10, 3, 10, 7},
        {12, 7, 10, 6, 7, 8, 10, 5, 10, 4, 4, 10, 8},
        {9, 6, 5, 9, 6, 9, 11, 11, 9, 8},
        {12, 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9},
        {12, 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6},
        {9, 6, 3, 11, 6, 5, 3, 5, 1, 3},
        {12, 1, 10, 2, 9, 11, 5, 9, 8, 11, 11, 6, 5},
        {15, 0, 3, 11, 0, 11, 6, 0, 6, 9, 5, 9, 6, 1, 10, 2},
        {15, 11, 5, 8, 11, 6, 5, 8, 5, 0, 10, 2, 5, 0, 5, 2},
        {12, 6, 3, 11, 6, 5, 3, 2, 3, 10, 10, 3, 5},
        {12, 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8},
        {9, 9, 6, 5, 9, 0, 6, 0, 2, 6},
        {15, 1, 8, 5, 1, 0, 8, 5, 8, 6, 3, 2, 8, 6, 8, 2},
        {6, 1, 6, 5, 2, 6, 1},
        {15, 1, 6, 3, 1, 10, 6, 3, 6, 8, 5, 9, 6, 8, 6, 9},
        {12, 10, 0, 1, 10, 6, 0, 9, 0, 5, 5, 0, 6},
        {6, 0, 8, 3, 5, 10, 6},
        {3, 10, 6, 5},
        {6, 11, 10, 5, 7, 11, 5},
        {9, 11, 10, 5, 11, 5, 7, 8, 0, 3},
        {9, 5, 7, 11, 5, 11, 10, 1, 0, 9},
        {12, 10, 5, 7, 10, 7, 11, 9, 1, 8, 8, 1, 3},
        {9, 11, 2, 1, 11, 1, 7, 7, 1, 5},
        {12, 0, 3, 8, 1, 7, 2, 1, 5, 7, 7, 11, 2},
        {12, 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11},
        {15, 7, 2, 5, 7, 11, 2, 5, 2, 9, 3, 8, 2, 9, 2, 8},
        {9, 2, 10, 5, 2, 5, 3, 3, 5, 7},
        {12, 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2},
        {12, 9, 1, 0, 5, 3, 10, 5, 7, 3, 3, 2, 10},
        {15, 9, 2, 8, 9, 1, 2, 8, 2, 7, 10, 5, 2, 7, 2, 5},
        {6, 1, 5, 3, 3, 5, 7},
        {9, 0, 7, 8, 0, 1, 7, 1, 5, 7},
        {9, 9, 3, 0, 9, 5, 3, 5, 7, 3},
        {6, 9, 7, 8, 5, 7, 9},
        {9, 5, 4, 8, 5, 8, 10, 10, 8, 11},
        {12, 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3},
        {12, 0, 9, 1, 8, 10, 4, 8, 11, 10, 10, 5, 4},
        {15, 10, 4, 11, 10, 5, 4, 11, 4, 3, 9, 1, 4, 3, 4, 1},
        {12, 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5},
        {15, 0, 11, 4, 0, 3, 11, 4, 11, 5, 2, 1, 11, 5, 11, 1},
        {15, 0, 5, 2, 0, 9, 5, 2, 5, 11, 4, 8, 5, 11, 5, 8},
        {6, 9, 5, 4, 2, 3, 11},
        {12, 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8},
        {9, 5, 2, 10, 5, 4, 2, 4, 0, 2},
        {15, 3, 2, 10, 3, 10, 5, 3, 5, 8, 4, 8, 5, 0, 9, 1},
        {12, 5, 2, 10, 5, 4, 2, 1, 2, 9, 9, 2, 4},
        {9, 8, 5, 4, 8, 3, 5, 3, 1, 5},
        {6, 0, 5, 4, 1, 5, 0},
        {12, 8, 5, 4, 8, 3, 5, 9, 5, 0, 0, 5, 3},
        {3, 9, 5, 4},
        {9, 4, 7, 11, 4, 11, 9, 9, 11, 10},
        {12, 0, 3, 8, 4, 7, 9, 9, 7, 11, 9, 11, 10},
        {12, 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4},
        {15, 3, 4, 1, 3, 8, 4, 1, 4, 10, 7, 11, 4, 10, 4, 11},
        {12, 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1},
        {15, 9, 4, 7, 9, 7, 11, 9, 11, 1, 2, 1, 11, 0, 3, 8},
        {9, 11, 4, 7, 11, 2, 4, 2, 0, 4},
        {12, 11, 4, 7, 11, 2, 4, 8, 4, 3, 3, 4, 2},
        {12, 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4},
        {15, 9, 7, 10, 9, 4, 7, 10, 7, 2, 8, 0, 7, 2, 7, 0},
        {15, 3, 10, 7, 3, 2, 10, 7, 10, 4, 1, 0, 10, 4, 10, 0},
        {6, 1, 2, 10, 8, 4, 7},
        {9, 4, 1, 9, 4, 7, 1, 7, 3, 1},
        {12, 4, 1, 9, 4, 7, 1, 0, 1, 8, 8, 1, 7},
        {6, 4, 3, 0, 7, 3, 4},
        {3, 4, 7, 8},
        {6, 9, 8, 10, 10, 8, 11},
        {9, 3, 9, 0, 3, 11, 9, 11, 10, 9},
        {9, 0, 10, 1, 0, 8, 10, 8, 11, 10},
        {6, 3, 10, 1, 11, 10, 3},
        {9, 1, 11, 2, 1, 9, 11, 9, 8, 11},
        {12, 3, 9, 0, 3, 11, 9, 1, 9, 2, 2, 9, 11},
        {6, 0, 11, 2, 8, 11, 0},
        {3, 3, 11, 2},
        {9, 2, 8, 3, 2, 10, 8, 10, 9, 8},
        {6, 9, 2, 10, 0, 2, 9},
        {12, 2, 8, 3, 2, 10, 8, 0, 8, 1, 1, 8, 10},
        {3, 1, 2, 10},
        {6, 1, 8, 3, 9, 8, 1},
        {3, 0, 1, 9},
        {3, 0, 8, 3},
        {0},
    };
    static_assert(std::size(TriangleTable) == 256);
} // namespace

namespace AIHoloImager
{
    class MarchingCubes::Impl
    {
    public:
        Impl(AIHoloImagerInternal& aihi) : gpu_system_(aihi.GpuSystemInstance())
        {
            edge_table_buff_ = GpuBuffer(gpu_system_, sizeof(EdgeTable), GpuHeap::Upload, GpuResourceFlag::None, L"edge_table_buff");
            {
                std::memcpy(edge_table_buff_.Map(), EdgeTable, sizeof(EdgeTable));
                edge_table_buff_.Unmap(GpuRange{0, edge_table_buff_.Size()});
            }
            edge_table_srv_ = GpuShaderResourceView(gpu_system_, edge_table_buff_, GpuFormat::R16_Uint);

            triangle_table_buff_ = GpuBuffer(gpu_system_, std::size(TriangleTable) * 16 * sizeof(uint16_t), GpuHeap::Upload,
                GpuResourceFlag::None, L"triangle_table_buff");
            {
                uint16_t* triangle_table_buff_ptr = triangle_table_buff_.Map<uint16_t>();
                for (size_t i = 0; i < std::size(TriangleTable); ++i)
                {
                    for (uint32_t j = 0; j < 16; ++j)
                    {
                        triangle_table_buff_ptr[i * 16 + j] = TriangleTable[i][j];
                    }
                }
                triangle_table_buff_.Unmap(GpuRange{0, triangle_table_buff_.Size()});
            }
            triangle_table_srv_ = GpuShaderResourceView(gpu_system_, triangle_table_buff_, GpuFormat::R16_Uint);

            {
                calc_cube_indices_cb_ = ConstantBuffer<CalcCubeIndicesConstantBuffer>(gpu_system_, L"calc_cube_indices_cb_");

                const ShaderInfo shader = {CalcCubeIndicesCs_shader, 1, 2, 2};
                calc_cube_indices_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                process_non_empty_cubes_cb_ =
                    ConstantBuffer<ProcessNonEmptyCubesConstantBuffer>(gpu_system_, L"process_non_empty_cubes_cb_");

                const ShaderInfo shader = {ProcessNonEmptyCubesCs_shader, 1, 4, 4};
                process_non_empty_cubes_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
            {
                gen_vertices_indices_cb_ = ConstantBuffer<GenVerticesIndicesConstantBuffer>(gpu_system_, L"gen_vertices_indices_cb_");

                const ShaderInfo shader = {GenVerticesIndicesCs_shader, 1, 7, 2};
                gen_vertices_indices_pipeline_ = GpuComputePipeline(gpu_system_, shader, {});
            }
        }

        Mesh Generate(const GpuTexture3D& scalar_deformation, float isovalue, float scale)
        {
            assert(scalar_deformation.Width(0) >= 3);

            const uint32_t size = scalar_deformation.Width(0);
            assert((scalar_deformation.Height(0) == size) && (scalar_deformation.Depth(0) == size));

            const uint32_t total_cubes = size * size * size;

            GpuShaderResourceView scalar_deformation_srv(gpu_system_, scalar_deformation);

            auto cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            // [num_non_empty_cubes, num_vertices, num_indices]
            GpuBuffer counter_buff(gpu_system_, 3 * sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, L"counter_buff");
            GpuUnorderedAccessView counter_uav(gpu_system_, counter_buff, GpuFormat::R32_Uint);
            const uint32_t zeros[] = {0, 0, 0, 0};
            cmd_list.Clear(counter_uav, zeros);

            GpuBuffer cube_offsets_buff(
                gpu_system_, total_cubes * sizeof(uint32_t), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, L"cube_offsets_buff");
            GpuShaderResourceView cube_offsets_srv(gpu_system_, cube_offsets_buff, GpuFormat::R32_Uint);
            {
                calc_cube_indices_cb_->size = size;
                calc_cube_indices_cb_->total_cubes = total_cubes;
                calc_cube_indices_cb_->isovalue = isovalue;
                calc_cube_indices_cb_.UploadToGpu();

                GpuUnorderedAccessView cube_offsets_uav(gpu_system_, cube_offsets_buff, GpuFormat::R32_Uint);

                constexpr uint32_t BlockDim = 256;

                const GeneralConstantBuffer* cbs[] = {&calc_cube_indices_cb_};
                const GpuShaderResourceView* srvs[] = {&edge_table_srv_, &scalar_deformation_srv};
                GpuUnorderedAccessView* uavs[] = {&cube_offsets_uav, &counter_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(calc_cube_indices_pipeline_, DivUp(total_cubes, BlockDim), 1, 1, shader_binding);
            }

            GpuReadBackBuffer counter_cpu_buff(gpu_system_, sizeof(glm::uvec2), L"counter_cpu_buff");
            cmd_list.Copy(counter_cpu_buff, 0, counter_buff, 0, sizeof(uint32_t));

            gpu_system_.Execute(std::move(cmd_list));
            gpu_system_.CpuWait();

            const uint32_t num_non_empty_cubes = *counter_cpu_buff.MappedData<uint32_t>();
            if (num_non_empty_cubes == 0)
            {
                return Mesh();
            }

            cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            GpuBuffer non_empty_cube_ids_buff(gpu_system_, num_non_empty_cubes * sizeof(uint32_t), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, L"non_empty_cube_ids_buff");
            GpuBuffer non_empty_cube_indices_buff(gpu_system_, num_non_empty_cubes * sizeof(uint32_t), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, L"non_empty_cube_indices_buff");
            GpuBuffer vertex_index_offsets_buff(gpu_system_, num_non_empty_cubes * sizeof(glm::uvec2), GpuHeap::Default,
                GpuResourceFlag::UnorderedAccess, L"vertex_index_offsets_buff");
            {
                process_non_empty_cubes_cb_->size = size;
                process_non_empty_cubes_cb_->total_cubes = total_cubes;
                process_non_empty_cubes_cb_->isovalue = isovalue;
                process_non_empty_cubes_cb_.UploadToGpu();

                GpuUnorderedAccessView non_empty_cube_ids_uav(gpu_system_, non_empty_cube_ids_buff, GpuFormat::R32_Uint);
                GpuUnorderedAccessView non_empty_cube_indices_uav(gpu_system_, non_empty_cube_indices_buff, GpuFormat::R32_Uint);
                GpuUnorderedAccessView vertex_index_offsets_uav(gpu_system_, vertex_index_offsets_buff, GpuFormat::RG32_Uint);

                constexpr uint32_t BlockDim = 256;

                const GeneralConstantBuffer* cbs[] = {&process_non_empty_cubes_cb_};
                const GpuShaderResourceView* srvs[] = {&edge_table_srv_, &triangle_table_srv_, &cube_offsets_srv, &scalar_deformation_srv};
                GpuUnorderedAccessView* uavs[] = {
                    &non_empty_cube_ids_uav, &non_empty_cube_indices_uav, &vertex_index_offsets_uav, &counter_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(process_non_empty_cubes_pipeline_, DivUp(total_cubes, BlockDim), 1, 1, shader_binding);
            }

            cmd_list.Copy(counter_cpu_buff, 0, counter_buff, sizeof(uint32_t), sizeof(glm::uvec2));

            gpu_system_.Execute(std::move(cmd_list));
            gpu_system_.CpuWait();

            const auto& counter = *counter_cpu_buff.MappedData<glm::uvec2>();
            const uint32_t num_vertices = counter.x;
            const uint32_t num_indices = counter.y;
            counter_cpu_buff.Reset();

            const VertexAttrib pos_only_vertex_attribs[] = {
                {VertexAttrib::Semantic::Position, 0, 3},
            };
            Mesh mesh(VertexDesc(pos_only_vertex_attribs), num_vertices, num_indices);

            cmd_list = gpu_system_.CreateCommandList(GpuSystem::CmdQueueType::Render);

            GpuReadBackBuffer mesh_vertices_cpu_buff(gpu_system_, num_vertices * sizeof(glm::vec3), L"mesh_vertices_cpu_buff");
            GpuReadBackBuffer mesh_indices_cpu_buff(gpu_system_, num_indices * sizeof(uint32_t), L"mesh_indices_cpu_buff");
            {
                gen_vertices_indices_cb_->size = size;
                gen_vertices_indices_cb_->num_non_empty_cubes = num_non_empty_cubes;
                gen_vertices_indices_cb_->isovalue = isovalue;
                gen_vertices_indices_cb_->scale = scale;
                gen_vertices_indices_cb_.UploadToGpu();

                GpuShaderResourceView non_empty_cube_ids_srv(gpu_system_, non_empty_cube_ids_buff, GpuFormat::R32_Uint);
                GpuShaderResourceView non_empty_cube_indices_srv(gpu_system_, non_empty_cube_indices_buff, GpuFormat::R32_Uint);
                GpuShaderResourceView vertex_index_offsets_srv(gpu_system_, vertex_index_offsets_buff, GpuFormat::RG32_Uint);

                GpuBuffer mesh_vertices_buff(
                    gpu_system_, mesh_vertices_cpu_buff.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, L"mesh_vertices_buff");
                GpuBuffer mesh_indices_buff(
                    gpu_system_, mesh_indices_cpu_buff.Size(), GpuHeap::Default, GpuResourceFlag::UnorderedAccess, L"mesh_indices_buff");
                GpuUnorderedAccessView mesh_vertices_uav(gpu_system_, mesh_vertices_buff, sizeof(glm::vec3));
                GpuUnorderedAccessView mesh_indices_uav(gpu_system_, mesh_indices_buff, GpuFormat::R32_Uint);

                constexpr uint32_t BlockDim = 256;

                const GeneralConstantBuffer* cbs[] = {&gen_vertices_indices_cb_};
                const GpuShaderResourceView* srvs[] = {&edge_table_srv_, &triangle_table_srv_, &scalar_deformation_srv,
                    &non_empty_cube_ids_srv, &non_empty_cube_indices_srv, &cube_offsets_srv, &vertex_index_offsets_srv};
                GpuUnorderedAccessView* uavs[] = {&mesh_vertices_uav, &mesh_indices_uav};
                const GpuCommandList::ShaderBinding shader_binding = {cbs, srvs, uavs};
                cmd_list.Compute(gen_vertices_indices_pipeline_, DivUp(num_non_empty_cubes, BlockDim), 1, 1, shader_binding);

                cmd_list.Copy(mesh_vertices_cpu_buff, mesh_vertices_buff);
                cmd_list.Copy(mesh_indices_cpu_buff, mesh_indices_buff);
            }
            vertex_index_offsets_buff.Reset();

            gpu_system_.Execute(std::move(cmd_list));
            gpu_system_.CpuWait();

            std::memcpy(mesh.VertexBuffer().data(), mesh_vertices_cpu_buff.MappedData(), mesh_vertices_cpu_buff.Size());
            std::memcpy(mesh.IndexBuffer().data(), mesh_indices_cpu_buff.MappedData(), mesh_indices_cpu_buff.Size());

            return mesh;
        }

    private:
        GpuSystem& gpu_system_;

        GpuBuffer edge_table_buff_;
        GpuShaderResourceView edge_table_srv_;

        GpuBuffer triangle_table_buff_;
        GpuShaderResourceView triangle_table_srv_;

        struct CalcCubeIndicesConstantBuffer
        {
            uint32_t size;
            uint32_t total_cubes;
            float isovalue;
            uint32_t padding;
        };
        ConstantBuffer<CalcCubeIndicesConstantBuffer> calc_cube_indices_cb_;
        GpuComputePipeline calc_cube_indices_pipeline_;

        struct ProcessNonEmptyCubesConstantBuffer
        {
            uint32_t size;
            uint32_t total_cubes;
            float isovalue;
            uint32_t padding;
        };
        ConstantBuffer<ProcessNonEmptyCubesConstantBuffer> process_non_empty_cubes_cb_;
        GpuComputePipeline process_non_empty_cubes_pipeline_;

        struct GenVerticesIndicesConstantBuffer
        {
            uint32_t size;
            uint32_t num_non_empty_cubes;
            float isovalue;
            float scale;
        };
        ConstantBuffer<GenVerticesIndicesConstantBuffer> gen_vertices_indices_cb_;
        GpuComputePipeline gen_vertices_indices_pipeline_;
    };

    MarchingCubes::MarchingCubes(AIHoloImagerInternal& aihi) : impl_(std::make_unique<Impl>(aihi))
    {
    }

    MarchingCubes::~MarchingCubes() noexcept = default;

    MarchingCubes::MarchingCubes(MarchingCubes&& other) noexcept = default;
    MarchingCubes& MarchingCubes::operator=(MarchingCubes&& other) noexcept = default;

    Mesh MarchingCubes::Generate(const GpuTexture3D& scalar_deformation, float isovalue, float scale)
    {
        return impl_->Generate(scalar_deformation, isovalue, scale);
    }
} // namespace AIHoloImager
