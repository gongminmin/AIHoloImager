// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include "Base/Symbol.hpp"

#ifdef AIHoloImagerGpuComputationLib_EXPORTS
    #define AIHI_GPU_COMPUTATION_LIB_API AIHI_SYMBOL_EXPORT
#else
    #define AIHI_GPU_COMPUTATION_LIB_API AIHI_SYMBOL_IMPORT
#endif
