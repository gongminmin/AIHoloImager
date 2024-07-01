// Copyright (c) 2024 Minmin Gong
//

#pragma once

#ifdef _WIN32
    #ifdef AIHoloImagerLib_EXPORTS
        #define AIHI_API __declspec(dllexport)
    #else
        #define AIHI_API __declspec(dllimport)
    #endif
#endif
