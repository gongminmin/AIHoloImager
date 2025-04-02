// Copyright (c) 2025 Minmin Gong
//

#pragma once

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

#include <directx/d3d12.h>
