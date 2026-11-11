# Copyright (c) 2026 Minmin Gong
#

set(aihi_gpu_gsplat_diff_render_files
    Include/GpuGSplatDiffRender/GpuGSplatDiffRender.hpp
    Include/GpuGSplatDiffRender/Symbol.hpp
    Source/Shader/BlendFloat3Cs.hlsl
    Source/Shader/BlendUnorm4Cs.hlsl
    Source/Shader/GSplatUtils.hlslh
    Source/Shader/PreprocessFwdCs.hlsl
    Source/Shader/RenderFwdGs.hlsl
    Source/Shader/RenderFwdPs.hlsl
    Source/Shader/RenderFwdVs.hlsl
    Source/Shader/Sh.hlslh
    Source/GpuGSplatDiffRender.cpp
    Source/pch.hpp
    Source/Torch/GpuGSplatDiffRenderTorch.cpp
    Source/Torch/GpuGSplatDiffRenderTorch.hpp
    Source/Torch/PyTorchBindings.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_gpu_gsplat_diff_render_files})

add_library(AIHoloImagerGpuGSplatDiffRender SHARED
    ${aihi_gpu_gsplat_diff_render_files}
)

AddShaderFile(Source/Shader/BlendFloat3Cs.hlsl "cs" "main")
AddShaderFile(Source/Shader/BlendUnorm4Cs.hlsl "cs" "main")
AddShaderFile(Source/Shader/PreprocessFwdCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/RenderFwdVs.hlsl "vs" "main")
AddShaderFile(Source/Shader/RenderFwdGs.hlsl "gs" "main")
AddShaderFile(Source/Shader/RenderFwdPs.hlsl "ps" "main")

set(torch_dir ${PROJECT_SOURCE_DIR}/External/Python/Venv/Venv/Lib/site-packages/torch)

target_include_directories(AIHoloImagerGpuGSplatDiffRender
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/Include

    PRIVATE
        ${torch_dir}/include
        ${torch_dir}/include/torch/csrc/api/include
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

target_precompile_headers(AIHoloImagerGpuGSplatDiffRender
    PRIVATE
        Source/pch.hpp
)

if(MSVC)
    target_compile_options(AIHoloImagerGpuGSplatDiffRender
        PRIVATE
            /wd4702 # Ignore unreachable code in PyTorch
    )
endif()

target_compile_definitions(AIHoloImagerGpuGSplatDiffRender
    PRIVATE
        Py_NO_LINK_LIB
        TORCH_API_INCLUDE_EXTENSION_H
)

target_link_directories(AIHoloImagerGpuGSplatDiffRender
    PRIVATE
        ${torch_dir}/lib
)

target_link_libraries(AIHoloImagerGpuGSplatDiffRender
    PRIVATE
        AIHoloImagerBase
        AIHoloImagerGpuSystem
        AIHoloImagerGpuComputationLib
        AIHoloImagerTensorConverter
        glm
        c10.lib
        torch_cpu.lib
        torch.lib
        torch_python.lib
        Python3::Python
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

add_dependencies(AIHoloImagerGpuGSplatDiffRender DeployPython)

set_target_properties(AIHoloImagerGpuGSplatDiffRender PROPERTIES FOLDER "Lib")
