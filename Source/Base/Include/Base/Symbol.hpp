// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include "Base/MiniWindows.hpp"

#ifdef _WIN32
    #define AIHI_SYMBOL_EXPORT __declspec(dllexport)
    #define AIHI_SYMBOL_IMPORT __declspec(dllimport)
#else
    #define AIHI_SYMBOL_EXPORT __attribute__((visibility("default")))
    #define AIHI_SYMBOL_IMPORT
#endif
