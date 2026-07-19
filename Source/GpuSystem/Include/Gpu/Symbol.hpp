// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include "Base/Symbol.hpp"

#ifdef AIHoloImagerGpuSystem_EXPORTS
    #define AIHI_GPU_SYS_API AIHI_SYMBOL_EXPORT
#else
    #define AIHI_GPU_SYS_API AIHI_SYMBOL_IMPORT
#endif
