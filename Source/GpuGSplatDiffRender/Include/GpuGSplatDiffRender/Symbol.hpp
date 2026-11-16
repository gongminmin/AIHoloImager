// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include "Base/Symbol.hpp"

#ifdef AIHoloImagerGpuGSplatDiffRender_EXPORTS
    #define AIHI_GPU_GSPLAT_DIFF_RENDER_API AIHI_SYMBOL_EXPORT
#else
    #define AIHI_GPU_GSPLAT_DIFF_RENDER_API AIHI_SYMBOL_IMPORT
#endif
