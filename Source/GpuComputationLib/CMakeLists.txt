# Copyright (c) 2026 Minmin Gong
#

set(aihi_gpu_computation_lib_files
    Include/GpuComputationLib/PrefixSumScanner/PrefixSumScanner.hpp
    Include/GpuComputationLib/Sorter/Sorter.hpp
    Include/GpuComputationLib/Symbol.hpp
    Source/PrefixSumScanner/Shader/ApplySumCs.hlslh
    Source/PrefixSumScanner/Shader/ApplySumUint2Cs.hlsl
    Source/PrefixSumScanner/Shader/ApplySumUintCs.hlsl
    Source/PrefixSumScanner/Shader/PrefixSumBlock.hlslh
    Source/PrefixSumScanner/Shader/PrefixSumCs.hlslh
    Source/PrefixSumScanner/Shader/PrefixSumUint2Cs.hlsl
    Source/PrefixSumScanner/Shader/PrefixSumUintCs.hlsl
    Source/PrefixSumScanner/PrefixSumScanner.cpp
    Source/Sorter/Shader/DownsweepCs.hlslh
    Source/Sorter/Shader/DownsweepCs_Uint_Uint.hlsl
    Source/Sorter/Shader/DownsweepCs_Uint2_Uint.hlsl
    Source/Sorter/Shader/ScanCs.hlslh
    Source/Sorter/Shader/ScanCs_Uint_Uint.hlsl
    Source/Sorter/Shader/ScanCs_Uint2_Uint.hlsl
    Source/Sorter/Shader/SortCommon.hlslh
    Source/Sorter/Shader/UpsweepCs.hlslh
    Source/Sorter/Shader/UpsweepCs_Uint_Uint.hlsl
    Source/Sorter/Shader/UpsweepCs_Uint2_Uint.hlsl
    Source/Sorter/Sorter.cpp
    Source/pch.hpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_gpu_computation_lib_files})

add_library(AIHoloImagerGpuComputationLib SHARED
    ${aihi_gpu_computation_lib_files}
)

AddShaderFile(Source/PrefixSumScanner/Shader/ApplySumUint2Cs.hlsl "cs" "main")
AddShaderFile(Source/PrefixSumScanner/Shader/ApplySumUintCs.hlsl "cs" "main")
AddShaderFile(Source/PrefixSumScanner/Shader/PrefixSumUint2Cs.hlsl "cs" "main")
AddShaderFile(Source/PrefixSumScanner/Shader/PrefixSumUintCs.hlsl "cs" "main")
AddShaderFile(Source/Sorter/Shader/DownsweepCs_Uint_Uint.hlsl "cs" "main")
AddShaderFile(Source/Sorter/Shader/DownsweepCs_Uint2_Uint.hlsl "cs" "main")
AddShaderFile(Source/Sorter/Shader/ScanCs_Uint_Uint.hlsl "cs" "main")
AddShaderFile(Source/Sorter/Shader/ScanCs_Uint2_Uint.hlsl "cs" "main")
AddShaderFile(Source/Sorter/Shader/UpsweepCs_Uint_Uint.hlsl "cs" "main")
AddShaderFile(Source/Sorter/Shader/UpsweepCs_Uint2_Uint.hlsl "cs" "main")

target_include_directories(AIHoloImagerGpuComputationLib
    PUBLIC
        Include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

target_precompile_headers(AIHoloImagerGpuComputationLib
    PRIVATE
        Source/pch.hpp
)

target_link_libraries(AIHoloImagerGpuComputationLib
    PRIVATE
        AIHoloImagerBase
        AIHoloImagerGpuSystem
)

set_target_properties(AIHoloImagerGpuComputationLib PROPERTIES FOLDER "Lib")
