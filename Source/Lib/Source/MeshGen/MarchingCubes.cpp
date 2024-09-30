// Copyright (c) 2024 Minmin Gong
//

#include "MarchingCubes.hpp"

#include <memory>
#include <tuple>
#include <vector>

#include <DirectXMath.h>

#include "Util/ErrorHandling.hpp"

using namespace DirectX;

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

    XMFLOAT3 InterpolateVertex(const XMFLOAT3& p0, const XMFLOAT3& p1, float v0, float v1, float isovalue)
    {
        const XMVECTOR vp0 = XMLoadFloat3(&p0);
        const XMVECTOR vp1 = XMLoadFloat3(&p1);

        XMVECTOR inter_p;
        if (v1 == v0)
        {
            inter_p = (vp0 + vp1) / 2;
        }
        else
        {
            inter_p = XMVectorLerp(vp0, vp1, (isovalue - v0) / (v1 - v0));
        }

        XMFLOAT3 ret;
        XMStoreFloat3(&ret, inter_p);

        return ret;
    }

    uint32_t PrefixSum(std::span<uint32_t> output, std::span<const uint32_t> input)
    {
        output[0] = 0;
        for (size_t i = 0; i < input.size() - 1; ++i)
        {
            output[i + 1] = input[i] + output[i];
        }
        return input.back() + output.back();
    }
} // namespace

namespace AIHoloImager
{
    Mesh MarchingCubes(std::span<const float> sdf, uint32_t grid_res, float isovalue)
    {
        assert(grid_res >= 2);

        const uint32_t size = grid_res + 1;
        assert(sdf.size() == size * size * size);

        const auto calc_offset = [size](uint32_t x, uint32_t y, uint32_t z) { return (x * size + y) * size + z; };
        const auto decompose_xyz = [grid_res](uint32_t index) {
            const uint32_t xy = index / grid_res;
            const uint32_t z = index - xy * grid_res;
            const uint32_t x = xy / grid_res;
            const uint32_t y = xy - x * grid_res;
            return std::make_tuple(x, y, z);
        };

        constexpr uint32_t OwnedEdges[] = {0, 3, 8};
        constexpr uint32_t XBoundaryOwnedEdges[] = {1, 9};
        constexpr uint32_t YBoundaryOwnedEdges[] = {2, 11};
        constexpr uint32_t ZBoundaryOwnedEdges[] = {4, 7};

        const uint32_t total_cubes = grid_res * grid_res * grid_res;

        std::vector<uint32_t> non_empty_cube_flags(total_cubes);
        std::vector<uint32_t> total_cube_indices(total_cubes);
        for (uint32_t cid = 0; cid < total_cubes; ++cid)
        {
            const auto [x, y, z] = decompose_xyz(cid);
            const float dist[] = {
                sdf[calc_offset(x + 0, y + 0, z + 0)],
                sdf[calc_offset(x + 1, y + 0, z + 0)],
                sdf[calc_offset(x + 1, y + 1, z + 0)],
                sdf[calc_offset(x + 0, y + 1, z + 0)],
                sdf[calc_offset(x + 0, y + 0, z + 1)],
                sdf[calc_offset(x + 1, y + 0, z + 1)],
                sdf[calc_offset(x + 1, y + 1, z + 1)],
                sdf[calc_offset(x + 0, y + 1, z + 1)],
            };

            uint32_t cube_index = 0;
            for (uint32_t m = 0; m < std::size(dist); ++m)
            {
                if (dist[m] <= isovalue)
                {
                    cube_index |= 1U << m;
                }
            }

            total_cube_indices[cid] = cube_index;
            non_empty_cube_flags[cid] = (EdgeTable[cube_index] != 0);
        }

        std::vector<uint32_t> non_empty_cube_offsets(total_cubes);
        const uint32_t num_non_empty_cubes = PrefixSum(non_empty_cube_offsets, non_empty_cube_flags);

        std::vector<uint32_t> non_empty_cube_ids(num_non_empty_cubes);
        std::vector<uint32_t> non_empty_cube_indices(num_non_empty_cubes);
        std::vector<uint32_t> non_empty_num_vertices(num_non_empty_cubes);
        std::vector<uint32_t> non_empty_num_indices(num_non_empty_cubes);
        for (uint32_t cid = 0; cid < total_cubes; ++cid)
        {
            if (non_empty_cube_flags[cid])
            {
                const uint32_t offset = non_empty_cube_offsets[cid];
                non_empty_cube_ids[offset] = cid;

                const uint32_t cube_index = total_cube_indices[cid];
                non_empty_cube_indices[offset] = cube_index;
                non_empty_num_indices[offset] = TriangleTable[cube_index][0];

                const uint32_t edges = EdgeTable[cube_index];
                uint32_t cube_num_vertices = 0;
                if (edges != 0)
                {
                    for (const uint32_t e : OwnedEdges)
                    {
                        if (edges & (1U << e))
                        {
                            ++cube_num_vertices;
                        }
                    }

                    const auto [x, y, z] = decompose_xyz(cid);
                    if (x == grid_res - 1)
                    {
                        for (const uint32_t e : XBoundaryOwnedEdges)
                        {
                            if (edges & (1U << e))
                            {
                                ++cube_num_vertices;
                            }
                        }
                    }
                    if (y == grid_res - 1)
                    {
                        for (const uint32_t e : YBoundaryOwnedEdges)
                        {
                            if (edges & (1U << e))
                            {
                                ++cube_num_vertices;
                            }
                        }
                    }
                    if (z == grid_res - 1)
                    {
                        for (const uint32_t e : ZBoundaryOwnedEdges)
                        {
                            if (edges & (1U << e))
                            {
                                ++cube_num_vertices;
                            }
                        }
                    }
                }
                non_empty_num_vertices[offset] = cube_num_vertices;
            }
        }

        std::vector<uint32_t> edges_offsets(num_non_empty_cubes);
        const uint32_t num_non_empty_edges = PrefixSum(edges_offsets, non_empty_num_vertices);

        std::vector<XMUINT2> non_empty_edges(num_non_empty_edges);
        for (uint32_t i = 0; i < num_non_empty_cubes; ++i)
        {
            const uint32_t cube_id = non_empty_cube_ids[i];
            const auto [x, y, z] = decompose_xyz(cube_id);
            const uint32_t cube_index = non_empty_cube_indices[i];
            const uint32_t edges = EdgeTable[cube_index];

            uint32_t offset = edges_offsets[i];
            for (const uint32_t e : OwnedEdges)
            {
                if (edges & (1U << e))
                {
                    non_empty_edges[offset] = XMUINT2(i, e);
                    ++offset;
                }
            }

            if (x == grid_res - 1)
            {
                for (const uint32_t e : XBoundaryOwnedEdges)
                {
                    if (edges & (1U << e))
                    {
                        non_empty_edges[offset] = XMUINT2(i, e);
                        ++offset;
                    }
                }
            }
            if (y == grid_res - 1)
            {
                for (const uint32_t e : YBoundaryOwnedEdges)
                {
                    if (edges & (1U << e))
                    {
                        non_empty_edges[offset] = XMUINT2(i, e);
                        ++offset;
                    }
                }
            }
            if (z == grid_res - 1)
            {
                for (const uint32_t e : ZBoundaryOwnedEdges)
                {
                    if (edges & (1U << e))
                    {
                        non_empty_edges[offset] = XMUINT2(i, e);
                        ++offset;
                    }
                }
            }
        }

        std::vector<XMFLOAT3> mesh_vertices(num_non_empty_edges);
        for (uint32_t ei = 0; ei < num_non_empty_edges; ++ei)
        {
            const uint32_t cube_id = non_empty_cube_ids[non_empty_edges[ei].x];
            const auto [i, j, k] = decompose_xyz(cube_id);
            const uint32_t edge_id = non_empty_edges[ei].y;

            const float x = static_cast<float>(i);
            const float y = static_cast<float>(j);
            const float z = static_cast<float>(k);
            const float x_dx = x + 1;
            const float y_dy = y + 1;
            const float z_dz = z + 1;

            const float dist[] = {
                sdf[calc_offset(i + 0, j + 0, k + 0)],
                sdf[calc_offset(i + 1, j + 0, k + 0)],
                sdf[calc_offset(i + 1, j + 1, k + 0)],
                sdf[calc_offset(i + 0, j + 1, k + 0)],
                sdf[calc_offset(i + 0, j + 0, k + 1)],
                sdf[calc_offset(i + 1, j + 0, k + 1)],
                sdf[calc_offset(i + 1, j + 1, k + 1)],
                sdf[calc_offset(i + 0, j + 1, k + 1)],
            };

            XMFLOAT3& inter_p = mesh_vertices[ei];
            switch (edge_id)
            {
            case 0:
                inter_p = InterpolateVertex(XMFLOAT3(x, y, z), XMFLOAT3(x_dx, y, z), dist[0], dist[1], isovalue);
                break;

            case 1:
                inter_p = InterpolateVertex(XMFLOAT3(x_dx, y, z), XMFLOAT3(x_dx, y_dy, z), dist[1], dist[2], isovalue);
                break;

            case 2:
                inter_p = InterpolateVertex(XMFLOAT3(x_dx, y_dy, z), XMFLOAT3(x, y_dy, z), dist[2], dist[3], isovalue);
                break;

            case 3:
                inter_p = InterpolateVertex(XMFLOAT3(x, y_dy, z), XMFLOAT3(x, y, z), dist[3], dist[0], isovalue);
                break;

            case 4:
                inter_p = InterpolateVertex(XMFLOAT3(x, y, z_dz), XMFLOAT3(x_dx, y, z_dz), dist[4], dist[5], isovalue);
                break;

            case 7:
                inter_p = InterpolateVertex(XMFLOAT3(x, y_dy, z_dz), XMFLOAT3(x, y, z_dz), dist[7], dist[4], isovalue);
                break;

            case 8:
                inter_p = InterpolateVertex(XMFLOAT3(x, y, z), XMFLOAT3(x, y, z_dz), dist[0], dist[4], isovalue);
                break;

            case 9:
                inter_p = InterpolateVertex(XMFLOAT3(x_dx, y, z), XMFLOAT3(x_dx, y, z_dz), dist[1], dist[5], isovalue);
                break;

            case 11:
                inter_p = InterpolateVertex(XMFLOAT3(x, y_dy, z), XMFLOAT3(x, y_dy, z_dz), dist[3], dist[7], isovalue);
                break;

            default:
                Unreachable();
            }
        }

        std::vector<uint32_t> index_offsets(num_non_empty_cubes);
        const uint32_t num_non_empty_indices = PrefixSum(index_offsets, non_empty_num_indices);

        std::vector<uint32_t> mesh_indices(num_non_empty_indices);
        for (uint32_t i = 0; i < num_non_empty_cubes; ++i)
        {
            const auto [x, y, z] = decompose_xyz(non_empty_cube_ids[i]);
            const uint32_t cube_index = non_empty_cube_indices[i];
            const uint32_t edges = EdgeTable[cube_index];

            uint32_t indices[12]{};

            uint32_t vertex_index = edges_offsets[i];
            for (const uint32_t e : OwnedEdges)
            {
                if (edges & (1U << e))
                {
                    indices[e] = vertex_index;
                    ++vertex_index;
                }
            }

            if (x == grid_res - 1)
            {
                for (const uint32_t e : XBoundaryOwnedEdges)
                {
                    if (edges & (1U << e))
                    {
                        indices[e] = vertex_index;
                        ++vertex_index;
                    }
                }
            }
            else
            {
                const uint32_t dx_cid = ((x + 1) * grid_res + y) * grid_res + z;
                if (non_empty_cube_flags[dx_cid])
                {
                    const uint32_t dx_ci = non_empty_cube_offsets[dx_cid];

                    const uint32_t dx_cube_index = non_empty_cube_indices[dx_ci];
                    const uint32_t dx_edges = EdgeTable[dx_cube_index];

                    uint32_t dx_vertex_index = edges_offsets[dx_ci];
                    if (dx_edges & (1U << 0))
                    {
                        ++dx_vertex_index;
                    }
                    if (dx_edges & (1U << 3))
                    {
                        indices[1] = dx_vertex_index;
                        ++dx_vertex_index;
                    }
                    if (dx_edges & (1U << 8))
                    {
                        indices[9] = dx_vertex_index;
                        ++dx_vertex_index;
                    }
                }
            }

            if (y == grid_res - 1)
            {
                for (const uint32_t e : YBoundaryOwnedEdges)
                {
                    if (edges & (1U << e))
                    {
                        indices[e] = vertex_index;
                        ++vertex_index;
                    }
                }
            }
            else
            {
                const uint32_t dy_cid = (x * grid_res + (y + 1)) * grid_res + z;
                if (non_empty_cube_flags[dy_cid])
                {
                    const uint32_t dy_ci = non_empty_cube_offsets[dy_cid];

                    const uint32_t dy_cube_index = non_empty_cube_indices[dy_ci];
                    const uint32_t dy_edges = EdgeTable[dy_cube_index];

                    uint32_t dy_vertex_index = edges_offsets[dy_ci];
                    if (dy_edges & (1U << 0))
                    {
                        indices[2] = dy_vertex_index;
                        ++dy_vertex_index;
                    }
                    if (dy_edges & (1U << 3))
                    {
                        ++dy_vertex_index;
                    }
                    if (dy_edges & (1U << 8))
                    {
                        indices[11] = dy_vertex_index;
                        ++dy_vertex_index;
                    }
                }
            }

            if (z == grid_res - 1)
            {
                for (const uint32_t e : ZBoundaryOwnedEdges)
                {
                    if (edges & (1U << e))
                    {
                        indices[e] = vertex_index;
                        ++vertex_index;
                    }
                }
            }
            else
            {
                const uint32_t dz_cid = (x * grid_res + y) * grid_res + (z + 1);
                if (non_empty_cube_flags[dz_cid])
                {
                    const uint32_t dz_ci = non_empty_cube_offsets[dz_cid];

                    const uint32_t dz_cube_index = non_empty_cube_indices[dz_ci];
                    const uint32_t dz_edges = EdgeTable[dz_cube_index];

                    uint32_t dz_vertex_index = edges_offsets[dz_ci];
                    if (dz_edges & (1U << 0))
                    {
                        indices[4] = dz_vertex_index;
                        ++dz_vertex_index;
                    }
                    if (dz_edges & (1U << 3))
                    {
                        indices[7] = dz_vertex_index;
                        ++dz_vertex_index;
                    }
                    if (dz_edges & (1U << 8))
                    {
                        ++dz_vertex_index;
                    }
                }
            }

            {
                const uint32_t dy_dz_cid = (x * grid_res + (y + 1)) * grid_res + (z + 1);
                if (non_empty_cube_flags[dy_dz_cid])
                {
                    const uint32_t dy_dz_ci = non_empty_cube_offsets[dy_dz_cid];

                    const uint32_t dy_dz_cube_index = non_empty_cube_indices[dy_dz_ci];
                    const uint32_t dy_dz_edges = EdgeTable[dy_dz_cube_index];
                    if (dy_dz_edges & (1U << 0))
                    {
                        indices[6] = edges_offsets[dy_dz_ci];
                    }
                }
            }
            {
                const uint32_t dx_dz_cid = ((x + 1) * grid_res + y) * grid_res + (z + 1);
                if (non_empty_cube_flags[dx_dz_cid])
                {
                    const uint32_t dx_dz_ci = non_empty_cube_offsets[dx_dz_cid];

                    const uint32_t dx_dz_cube_index = non_empty_cube_indices[dx_dz_ci];
                    const uint32_t dx_dz_edges = EdgeTable[dx_dz_cube_index];
                    if (dx_dz_edges & (1U << 3))
                    {
                        indices[5] = edges_offsets[dx_dz_ci] + ((dx_dz_edges & (1U << 0)) ? 1 : 0);
                    }
                }
            }
            {
                const uint32_t dx_dy_cid = ((x + 1) * grid_res + (y + 1)) * grid_res + z;
                if (non_empty_cube_flags[dx_dy_cid])
                {
                    const uint32_t dx_dy_ci = non_empty_cube_offsets[dx_dy_cid];

                    const uint32_t dx_dy_cube_index = non_empty_cube_indices[dx_dy_ci];
                    const uint32_t dx_dy_edges = EdgeTable[dx_dy_cube_index];
                    if (dx_dy_edges & (1U << 8))
                    {
                        indices[10] = edges_offsets[dx_dy_ci] + ((dx_dy_edges & (1U << 0)) ? 1 : 0) + ((dx_dy_edges & (1U << 3)) ? 1 : 0);
                    }
                }
            }

            const uint32_t offset = index_offsets[i];
            const uint8_t* triangle_table_row = &TriangleTable[cube_index][1];
            for (uint32_t m = 0; m < TriangleTable[cube_index][0]; ++m)
            {
                mesh_indices[offset + m] = indices[triangle_table_row[m]];
            }
        }

        const VertexAttrib pos_only_vertex_attribs[] = {
            {VertexAttrib::Semantic::Position, 0, 3},
        };
        Mesh mesh(VertexDesc(pos_only_vertex_attribs), num_non_empty_edges, num_non_empty_indices);

        mesh.VertexBuffer(&mesh_vertices[0].x);
        mesh.Indices(mesh_indices);

        return mesh;
    }
} // namespace AIHoloImager
