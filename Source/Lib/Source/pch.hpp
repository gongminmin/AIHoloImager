// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <format>
#include <memory>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

#include <DirectXMath.h>
#include <assimp/types.h>
#include <directx/d3d12.h>
#include <third_party/eigen/Eigen/Core>
