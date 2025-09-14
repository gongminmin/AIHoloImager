# Copyright (c) 2025 Minmin Gong
#

set(aihi_gpu_diff_render_files
    Source/Shader/AntiAliasBwdCs.hlsl
    Source/Shader/AntiAliasConstructOppoVertCs.hlsl
    Source/Shader/AntiAliasConstructOppoVertHashCs.hlsl
    Source/Shader/AntiAliasFwdCs.hlsl
    Source/Shader/AntiAliasIndirectCs.hlsl
    Source/Shader/Common.hlslh
    Source/Shader/InterpolateBwdCs.hlsl
    Source/Shader/InterpolateFwdCs.hlsl
    Source/Shader/RasterizeBwdCs.hlsl
    Source/Shader/RasterizeFwdGs.hlsl
    Source/Shader/RasterizeFwdPs.hlsl
    Source/Shader/RasterizeFwdVs.hlsl
    Source/Shader/TextureBwdCs.hlsl
    Source/Shader/TextureCopyCs.hlsl
    Source/Shader/TextureFwdCs.hlsl
    Source/pch.hpp
    Source/GpuDiffRender.cpp
    Source/GpuDiffRender.hpp
    Source/Torch/GpuDiffRenderTorch.cpp
    Source/Torch/GpuDiffRenderTorch.hpp
    Source/Torch/PyTorchBindings.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_gpu_diff_render_files})

add_library(AIHoloImagerGpuDiffRender SHARED
    ${aihi_gpu_diff_render_files}
)

AddShaderFile(Source/Shader/RasterizeFwdVs.hlsl "vs" "main")
AddShaderFile(Source/Shader/RasterizeFwdGs.hlsl "gs" "main")
AddShaderFile(Source/Shader/RasterizeFwdPs.hlsl "ps" "main")
AddShaderFile(Source/Shader/RasterizeBwdCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/InterpolateFwdCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/InterpolateBwdCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/AntiAliasIndirectCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/AntiAliasFwdCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/AntiAliasBwdCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/AntiAliasConstructOppoVertCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/AntiAliasConstructOppoVertHashCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/TextureCopyCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/TextureFwdCs.hlsl "cs" "main")
AddShaderFile(Source/Shader/TextureBwdCs.hlsl "cs" "main")

set(torch_dir ${PROJECT_SOURCE_DIR}/External/PythonVenv/PythonVenv/Lib/site-packages/torch)

target_include_directories(AIHoloImagerGpuDiffRender
    PRIVATE
        ${torch_dir}/include
        ${torch_dir}/include/torch/csrc/api/include
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

target_precompile_headers(AIHoloImagerGpuDiffRender
    PRIVATE
        Source/pch.hpp
)

if(MSVC)
    target_compile_options(AIHoloImagerGpuDiffRender
        PRIVATE
            /wd4702 # Ignore unreachable code in PyTorch
    )
endif()

target_compile_definitions(AIHoloImagerGpuDiffRender
    PRIVATE
        TORCH_API_INCLUDE_EXTENSION_H
)

target_link_directories(AIHoloImagerGpuDiffRender
    PRIVATE
        ${torch_dir}/lib
)

target_link_libraries(AIHoloImagerGpuDiffRender
    PRIVATE
        AIHoloImagerBase
        AIHoloImagerGpuSystem
        AIHoloImagerTensorConverter
        glm
        c10.lib
        torch_cpu.lib
        torch.lib
        torch_python.lib
        Python3::Python
)

set_target_properties(AIHoloImagerGpuDiffRender
    PROPERTIES
        SUFFIX ".pyd"
        POSTFIX ""
        DEBUG_POSTFIX ""
        MINSIZEREL_POSTFIX ""
        RELWITHDEBINFO_POSTFIX ""
        RELEASE_POSTFIX ""
)

add_dependencies(AIHoloImagerGpuDiffRender DeployPython)
