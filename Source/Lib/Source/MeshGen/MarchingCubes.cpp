// Copyright (c) 2024 Minmin Gong
//

#include "MarchingCubes.hpp"

#include <memory>
#include <vector>

#include <DirectXMath.h>

using namespace DirectX;

// Based on https://github.com/pmneila/PyMCubes/tree/master/mcubes/src

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

    constexpr int8_t TriangleTable[][16] = {
        {},
        {0, 8, 3},
        {0, 1, 9},
        {1, 8, 3, 9, 8, 1},
        {1, 2, 10},
        {0, 8, 3, 1, 2, 10},
        {9, 2, 10, 0, 2, 9},
        {2, 8, 3, 2, 10, 8, 10, 9, 8},
        {3, 11, 2},
        {0, 11, 2, 8, 11, 0},
        {1, 9, 0, 2, 3, 11},
        {1, 11, 2, 1, 9, 11, 9, 8, 11},
        {3, 10, 1, 11, 10, 3},
        {0, 10, 1, 0, 8, 10, 8, 11, 10},
        {3, 9, 0, 3, 11, 9, 11, 10, 9},
        {9, 8, 10, 10, 8, 11},
        {4, 7, 8},
        {4, 3, 0, 7, 3, 4},
        {0, 1, 9, 8, 4, 7},
        {4, 1, 9, 4, 7, 1, 7, 3, 1},
        {1, 2, 10, 8, 4, 7},
        {3, 4, 7, 3, 0, 4, 1, 2, 10},
        {9, 2, 10, 9, 0, 2, 8, 4, 7},
        {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4},
        {8, 4, 7, 3, 11, 2},
        {11, 4, 7, 11, 2, 4, 2, 0, 4},
        {9, 0, 1, 8, 4, 7, 2, 3, 11},
        {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1},
        {3, 10, 1, 3, 11, 10, 7, 8, 4},
        {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4},
        {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3},
        {4, 7, 11, 4, 11, 9, 9, 11, 10},
        {9, 5, 4},
        {9, 5, 4, 0, 8, 3},
        {0, 5, 4, 1, 5, 0},
        {8, 5, 4, 8, 3, 5, 3, 1, 5},
        {1, 2, 10, 9, 5, 4},
        {3, 0, 8, 1, 2, 10, 4, 9, 5},
        {5, 2, 10, 5, 4, 2, 4, 0, 2},
        {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8},
        {9, 5, 4, 2, 3, 11},
        {0, 11, 2, 0, 8, 11, 4, 9, 5},
        {0, 5, 4, 0, 1, 5, 2, 3, 11},
        {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5},
        {10, 3, 11, 10, 1, 3, 9, 5, 4},
        {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10},
        {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3},
        {5, 4, 8, 5, 8, 10, 10, 8, 11},
        {9, 7, 8, 5, 7, 9},
        {9, 3, 0, 9, 5, 3, 5, 7, 3},
        {0, 7, 8, 0, 1, 7, 1, 5, 7},
        {1, 5, 3, 3, 5, 7},
        {9, 7, 8, 9, 5, 7, 10, 1, 2},
        {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3},
        {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2},
        {2, 10, 5, 2, 5, 3, 3, 5, 7},
        {7, 9, 5, 7, 8, 9, 3, 11, 2},
        {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11},
        {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7},
        {11, 2, 1, 11, 1, 7, 7, 1, 5},
        {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11},
        {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0},
        {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0},
        {11, 10, 5, 7, 11, 5},
        {10, 6, 5},
        {0, 8, 3, 5, 10, 6},
        {9, 0, 1, 5, 10, 6},
        {1, 8, 3, 1, 9, 8, 5, 10, 6},
        {1, 6, 5, 2, 6, 1},
        {1, 6, 5, 1, 2, 6, 3, 0, 8},
        {9, 6, 5, 9, 0, 6, 0, 2, 6},
        {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8},
        {2, 3, 11, 10, 6, 5},
        {11, 0, 8, 11, 2, 0, 10, 6, 5},
        {0, 1, 9, 2, 3, 11, 5, 10, 6},
        {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11},
        {6, 3, 11, 6, 5, 3, 5, 1, 3},
        {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6},
        {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9},
        {6, 5, 9, 6, 9, 11, 11, 9, 8},
        {5, 10, 6, 4, 7, 8},
        {4, 3, 0, 4, 7, 3, 6, 5, 10},
        {1, 9, 0, 5, 10, 6, 8, 4, 7},
        {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4},
        {6, 1, 2, 6, 5, 1, 4, 7, 8},
        {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7},
        {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6},
        {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9},
        {3, 11, 2, 7, 8, 4, 10, 6, 5},
        {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11},
        {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6},
        {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6},
        {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6},
        {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11},
        {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7},
        {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9},
        {10, 4, 9, 6, 4, 10},
        {4, 10, 6, 4, 9, 10, 0, 8, 3},
        {10, 0, 1, 10, 6, 0, 6, 4, 0},
        {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10},
        {1, 4, 9, 1, 2, 4, 2, 6, 4},
        {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4},
        {0, 2, 4, 4, 2, 6},
        {8, 3, 2, 8, 2, 4, 4, 2, 6},
        {10, 4, 9, 10, 6, 4, 11, 2, 3},
        {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6},
        {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10},
        {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1},
        {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3},
        {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1},
        {3, 11, 6, 3, 6, 0, 0, 6, 4},
        {6, 4, 8, 11, 6, 8},
        {7, 10, 6, 7, 8, 10, 8, 9, 10},
        {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10},
        {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0},
        {10, 6, 7, 10, 7, 1, 1, 7, 3},
        {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7},
        {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9},
        {7, 8, 0, 7, 0, 6, 6, 0, 2},
        {7, 3, 2, 6, 7, 2},
        {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7},
        {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7},
        {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11},
        {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1},
        {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6},
        {0, 9, 1, 11, 6, 7},
        {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0},
        {7, 11, 6},
        {7, 6, 11},
        {3, 0, 8, 11, 7, 6},
        {0, 1, 9, 11, 7, 6},
        {8, 1, 9, 8, 3, 1, 11, 7, 6},
        {10, 1, 2, 6, 11, 7},
        {1, 2, 10, 3, 0, 8, 6, 11, 7},
        {2, 9, 0, 2, 10, 9, 6, 11, 7},
        {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8},
        {7, 2, 3, 6, 2, 7},
        {7, 0, 8, 7, 6, 0, 6, 2, 0},
        {2, 7, 6, 2, 3, 7, 0, 1, 9},
        {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6},
        {10, 7, 6, 10, 1, 7, 1, 3, 7},
        {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8},
        {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7},
        {7, 6, 10, 7, 10, 8, 8, 10, 9},
        {6, 8, 4, 11, 8, 6},
        {3, 6, 11, 3, 0, 6, 0, 4, 6},
        {8, 6, 11, 8, 4, 6, 9, 0, 1},
        {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6},
        {6, 8, 4, 6, 11, 8, 2, 10, 1},
        {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6},
        {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9},
        {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3},
        {8, 2, 3, 8, 4, 2, 4, 6, 2},
        {0, 4, 2, 4, 6, 2},
        {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8},
        {1, 9, 4, 1, 4, 2, 2, 4, 6},
        {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1},
        {10, 1, 0, 10, 0, 6, 6, 0, 4},
        {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3},
        {10, 9, 4, 6, 10, 4},
        {4, 9, 5, 7, 6, 11},
        {0, 8, 3, 4, 9, 5, 11, 7, 6},
        {5, 0, 1, 5, 4, 0, 7, 6, 11},
        {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5},
        {9, 5, 4, 10, 1, 2, 7, 6, 11},
        {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5},
        {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2},
        {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6},
        {7, 2, 3, 7, 6, 2, 5, 4, 9},
        {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7},
        {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0},
        {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8},
        {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7},
        {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4},
        {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10},
        {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10},
        {6, 9, 5, 6, 11, 9, 11, 8, 9},
        {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5},
        {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11},
        {6, 11, 3, 6, 3, 5, 5, 3, 1},
        {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6},
        {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10},
        {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5},
        {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3},
        {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2},
        {9, 5, 6, 9, 6, 0, 0, 6, 2},
        {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8},
        {1, 5, 6, 2, 1, 6},
        {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6},
        {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0},
        {0, 3, 8, 5, 6, 10},
        {10, 5, 6},
        {11, 5, 10, 7, 5, 11},
        {11, 5, 10, 11, 7, 5, 8, 3, 0},
        {5, 11, 7, 5, 10, 11, 1, 9, 0},
        {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1},
        {11, 1, 2, 11, 7, 1, 7, 5, 1},
        {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11},
        {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7},
        {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2},
        {2, 5, 10, 2, 3, 5, 3, 7, 5},
        {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5},
        {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2},
        {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2},
        {1, 3, 5, 3, 7, 5},
        {0, 8, 7, 0, 7, 1, 1, 7, 5},
        {9, 0, 3, 9, 3, 5, 5, 3, 7},
        {9, 8, 7, 5, 9, 7},
        {5, 8, 4, 5, 10, 8, 10, 11, 8},
        {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0},
        {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5},
        {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4},
        {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8},
        {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11},
        {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5},
        {9, 4, 5, 2, 11, 3},
        {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4},
        {5, 10, 2, 5, 2, 4, 4, 2, 0},
        {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9},
        {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2},
        {8, 4, 5, 8, 5, 3, 3, 5, 1},
        {0, 4, 5, 1, 0, 5},
        {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5},
        {9, 4, 5},
        {4, 11, 7, 4, 9, 11, 9, 10, 11},
        {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11},
        {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11},
        {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4},
        {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2},
        {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3},
        {11, 7, 4, 11, 4, 2, 2, 4, 0},
        {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4},
        {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9},
        {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7},
        {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10},
        {1, 10, 2, 8, 7, 4},
        {4, 9, 1, 4, 1, 7, 7, 1, 3},
        {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1},
        {4, 0, 3, 7, 4, 3},
        {4, 8, 7},
        {9, 10, 8, 10, 11, 8},
        {3, 0, 9, 3, 9, 11, 11, 9, 10},
        {0, 1, 10, 0, 10, 8, 8, 10, 11},
        {3, 1, 10, 11, 3, 10},
        {1, 2, 11, 1, 11, 9, 9, 11, 8},
        {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9},
        {0, 2, 11, 8, 0, 11},
        {3, 2, 11},
        {2, 3, 8, 2, 8, 10, 10, 8, 9},
        {9, 10, 2, 0, 9, 2},
        {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8},
        {1, 10, 2},
        {1, 3, 8, 9, 1, 8},
        {0, 9, 1},
        {0, 3, 8},
        {},
    };
    static_assert(std::size(TriangleTable) == 256);

    constexpr uint8_t NumIndicesTable[] = {
        0,
        3,
        3,
        6,
        3,
        6,
        6,
        9,
        3,
        6,
        6,
        9,
        6,
        9,
        9,
        6,
        3,
        6,
        6,
        9,
        6,
        9,
        9,
        12,
        6,
        9,
        9,
        12,
        9,
        12,
        12,
        9,
        3,
        6,
        6,
        9,
        6,
        9,
        9,
        12,
        6,
        9,
        9,
        12,
        9,
        12,
        12,
        9,
        6,
        9,
        9,
        6,
        9,
        12,
        12,
        9,
        9,
        12,
        12,
        9,
        12,
        15,
        15,
        6,
        3,
        6,
        6,
        9,
        6,
        9,
        9,
        12,
        6,
        9,
        9,
        12,
        9,
        12,
        12,
        9,
        6,
        9,
        9,
        12,
        9,
        12,
        12,
        15,
        9,
        12,
        12,
        15,
        12,
        15,
        15,
        12,
        6,
        9,
        9,
        12,
        9,
        12,
        6,
        9,
        9,
        12,
        12,
        15,
        12,
        15,
        9,
        6,
        9,
        12,
        12,
        9,
        12,
        15,
        9,
        6,
        12,
        15,
        15,
        12,
        15,
        6,
        12,
        3,
        3,
        6,
        6,
        9,
        6,
        9,
        9,
        12,
        6,
        9,
        9,
        12,
        9,
        12,
        12,
        9,
        6,
        9,
        9,
        12,
        9,
        12,
        12,
        15,
        9,
        6,
        12,
        9,
        12,
        9,
        15,
        6,
        6,
        9,
        9,
        12,
        9,
        12,
        12,
        15,
        9,
        12,
        12,
        15,
        12,
        15,
        15,
        12,
        9,
        12,
        12,
        9,
        12,
        15,
        15,
        12,
        12,
        9,
        15,
        6,
        15,
        12,
        6,
        3,
        6,
        9,
        9,
        12,
        9,
        12,
        12,
        15,
        9,
        12,
        12,
        15,
        6,
        9,
        9,
        6,
        9,
        12,
        12,
        15,
        12,
        15,
        15,
        6,
        12,
        9,
        15,
        12,
        9,
        6,
        12,
        3,
        9,
        12,
        12,
        15,
        12,
        15,
        9,
        12,
        12,
        15,
        15,
        6,
        9,
        12,
        6,
        3,
        6,
        9,
        9,
        6,
        9,
        12,
        6,
        3,
        9,
        6,
        12,
        3,
        6,
        3,
        3,
        0,
    };
    static_assert(std::size(NumIndicesTable) == 256);

    uint32_t AddVertex(const XMFLOAT3& p0, const XMFLOAT3& p1, float v0, float v1, float isovalue, std::vector<XMFLOAT3>& vertices)
    {
        const uint32_t index = static_cast<uint32_t>(vertices.size());

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
        XMStoreFloat3(&vertices.emplace_back(), inter_p);

        return index;
    }
} // namespace

namespace AIHoloImager
{
    Mesh MarchingCubes(std::span<const float> sdf, uint32_t grid_res, float isovalue)
    {
        const uint32_t size = grid_res + 1;

        assert(sdf.size() == size * size * size);
        assert(grid_res >= 2);

        auto shared_indices = std::make_unique<XMUINT3[]>(2 * size * size);
        const auto calc_offset = [size](uint32_t x, uint32_t y, uint32_t z) { return (x * size + y) * size + z; };

        std::vector<XMFLOAT3> mesh_vertices;
        std::vector<uint32_t> mesh_indices;
        for (uint32_t i = 0; i < grid_res; ++i)
        {
            const uint32_t i_mod_2 = i & 1U;
            const uint32_t i_mod_2_inv = (i_mod_2 ? 0 : 1);

            const float x = static_cast<float>(i);
            const float x_dx = x + 1;

            for (uint32_t j = 0; j < grid_res; ++j)
            {
                const float y = static_cast<float>(j);
                const float y_dy = y + 1;

                for (uint32_t k = 0; k < grid_res; ++k)
                {
                    const float z = static_cast<float>(k);
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

                    uint32_t cube_index = 0;
                    for (uint32_t m = 0; m < std::size(dist); ++m)
                    {
                        if (dist[m] <= isovalue)
                        {
                            cube_index |= 1 << m;
                        }
                    }

                    const uint32_t edges = EdgeTable[cube_index];
                    uint32_t indices[12]{};

                    if (edges & (1U << 6))
                    {
                        uint32_t& index = shared_indices[calc_offset(i_mod_2_inv, j + 1, k + 1)].x;
                        index = AddVertex(XMFLOAT3(x_dx, y_dy, z_dz), XMFLOAT3(x, y_dy, z_dz), dist[6], dist[7], isovalue, mesh_vertices);
                        indices[6] = index;
                    }
                    if (edges & (1U << 5))
                    {
                        uint32_t& index = shared_indices[calc_offset(i_mod_2_inv, j + 1, k + 1)].y;
                        index = AddVertex(XMFLOAT3(x_dx, y, z_dz), XMFLOAT3(x_dx, y_dy, z_dz), dist[5], dist[6], isovalue, mesh_vertices);
                        indices[5] = index;
                    }
                    if (edges & (1U << 10))
                    {
                        uint32_t& index = shared_indices[calc_offset(i_mod_2_inv, j + 1, k + 1)].z;
                        index = AddVertex(XMFLOAT3(x_dx, y_dy, z), XMFLOAT3(x_dx, y_dy, z_dz), dist[2], dist[6], isovalue, mesh_vertices);
                        indices[10] = index;
                    }

                    if (edges & (1U << 0))
                    {
                        uint32_t index;
                        if ((j == 0) && (k == 0))
                        {
                            index = AddVertex(XMFLOAT3(x, y, z), XMFLOAT3(x_dx, y, z), dist[0], dist[1], isovalue, mesh_vertices);
                        }
                        else
                        {
                            index = shared_indices[calc_offset(i_mod_2_inv, j, k)].x;
                        }
                        indices[0] = index;
                    }
                    if (edges & (1U << 1))
                    {
                        uint32_t& index = shared_indices[calc_offset(i_mod_2_inv, j + 1, k)].y;
                        if (k == 0)
                        {
                            index = AddVertex(XMFLOAT3(x_dx, y, z), XMFLOAT3(x_dx, y_dy, z), dist[1], dist[2], isovalue, mesh_vertices);
                        }
                        indices[1] = index;
                    }
                    if (edges & (1U << 2))
                    {
                        uint32_t& index = shared_indices[calc_offset(i_mod_2_inv, j + 1, k)].x;
                        if (k == 0)
                        {
                            index = AddVertex(XMFLOAT3(x_dx, y_dy, z), XMFLOAT3(x, y_dy, z), dist[2], dist[3], isovalue, mesh_vertices);
                        }
                        indices[2] = index;
                    }
                    if (edges & (1U << 3))
                    {
                        uint32_t index;
                        if ((i == 0) && (k == 0))
                        {
                            index = AddVertex(XMFLOAT3(x, y_dy, z), XMFLOAT3(x, y, z), dist[3], dist[0], isovalue, mesh_vertices);
                        }
                        else
                        {
                            index = shared_indices[calc_offset(i_mod_2, j + 1, k)].y;
                        }
                        indices[3] = index;
                    }
                    if (edges & (1U << 4))
                    {
                        uint32_t& index = shared_indices[calc_offset(i_mod_2_inv, j, k + 1)].x;
                        if (j == 0)
                        {
                            index = AddVertex(XMFLOAT3(x, y, z_dz), XMFLOAT3(x_dx, y, z_dz), dist[4], dist[5], isovalue, mesh_vertices);
                        }
                        indices[4] = index;
                    }
                    if (edges & (1U << 7))
                    {
                        uint32_t& index = shared_indices[calc_offset(i_mod_2, j + 1, k + 1)].y;
                        if (i == 0)
                        {
                            index = AddVertex(XMFLOAT3(x, y_dy, z_dz), XMFLOAT3(x, y, z_dz), dist[7], dist[4], isovalue, mesh_vertices);
                        }
                        indices[7] = index;
                    }
                    if (edges & (1U << 8))
                    {
                        uint32_t index;
                        if ((i == 0) && (j == 0))
                        {
                            index = AddVertex(XMFLOAT3(x, y, z), XMFLOAT3(x, y, z_dz), dist[0], dist[4], isovalue, mesh_vertices);
                        }
                        else
                        {
                            index = shared_indices[calc_offset(i_mod_2, j, k + 1)].z;
                        }
                        indices[8] = index;
                    }
                    if (edges & (1U << 9))
                    {
                        uint32_t& index = shared_indices[calc_offset(i_mod_2_inv, j, k + 1)].z;
                        if (j == 0)
                        {
                            index = AddVertex(XMFLOAT3(x_dx, y, z), XMFLOAT3(x_dx, y, z_dz), dist[1], dist[5], isovalue, mesh_vertices);
                        }
                        indices[9] = index;
                    }
                    if (edges & (1U << 11))
                    {
                        uint32_t& index = shared_indices[calc_offset(i_mod_2, j + 1, k + 1)].z;
                        if (i == 0)
                        {
                            index = AddVertex(XMFLOAT3(x, y_dy, z), XMFLOAT3(x, y_dy, z_dz), dist[3], dist[7], isovalue, mesh_vertices);
                        }
                        indices[11] = index;
                    }

                    const int8_t* triangle_table_row = TriangleTable[cube_index];
                    for (uint32_t m = 0; m < NumIndicesTable[cube_index]; m += 3)
                    {
                        mesh_indices.insert(mesh_indices.end(), {
                                                                    indices[triangle_table_row[m + 0]],
                                                                    indices[triangle_table_row[m + 2]], // Flip the triangle
                                                                    indices[triangle_table_row[m + 1]],
                                                                });
                    }
                }
            }
        }

        const VertexAttrib pos_only_vertex_attribs[] = {
            {VertexAttrib::Semantic::Position, 0, 3},
        };
        Mesh mesh(
            VertexDesc(pos_only_vertex_attribs), static_cast<uint32_t>(mesh_vertices.size()), static_cast<uint32_t>(mesh_indices.size()));

        mesh.VertexBuffer(&mesh_vertices[0].x);
        mesh.Indices(mesh_indices);

        return mesh;
    }
} // namespace AIHoloImager
