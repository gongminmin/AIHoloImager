# Copyright (c) 2026 Minmin Gong
#

set(aihi_gpu_gsplat_diff_render_files
    Include/GpuGSplatDiffRender/GpuGSplatDiffRender.hpp
    Include/GpuGSplatDiffRender/Symbol.hpp
    Source/Shader/BlendCs.hlsl
    Source/Shader/PreprocessFwdCs.hlsl
    Source/Shader/RenderFwdGs.hlsl
    Source/Shader/RenderFwdPs.hlsl
    Source/Shader/RenderFwdVs.hlsl
    Source/Shader/Sh.hlslh
    Source/GpuGSplatDiffRender.cpp
    Source/pch.hpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_gpu_gsplat_diff_render_files})

add_library(AIHoloImagerGpuGSplatDiffRender SHARED
    ${aihi_gpu_gsplat_diff_render_files}
)

AddShaderFile(Source/Shader/BlendCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/PreprocessFwdCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/RenderFwdVs.hlsl "vs" "main")
AddShaderFile(Source/Shader/RenderFwdGs.hlsl "gs" "main")
AddShaderFile(Source/Shader/RenderFwdPs.hlsl "ps" "main")

target_include_directories(AIHoloImagerGpuGSplatDiffRender
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/Include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

target_precompile_headers(AIHoloImagerGpuGSplatDiffRender
    PRIVATE
        Source/pch.hpp
)

target_compile_definitions(AIHoloImagerGpuGSplatDiffRender
    PRIVATE
        Py_NO_LINK_LIB
)

target_link_libraries(AIHoloImagerGpuGSplatDiffRender
    PRIVATE
        AIHoloImagerBase
        AIHoloImagerGpuSystem
        AIHoloImagerGpuComputationLib
        glm
)

set_target_properties(AIHoloImagerGpuGSplatDiffRender
    PROPERTIES
        SUFFIX ".pyd"
        POSTFIX ""
        DEBUG_POSTFIX ""
        MINSIZEREL_POSTFIX ""
        RELWITHDEBINFO_POSTFIX ""
        RELEASE_POSTFIX ""
)

set_target_properties(AIHoloImagerGpuGSplatDiffRender PROPERTIES FOLDER "Lib")
