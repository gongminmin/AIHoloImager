# Copyright (c) 2025 Minmin Gong
#

set(aihi_subm_conv_files
    Source/pch.hpp
    Source/Torch/Shader/BuildCoordHashCs.hlsl
    Source/Torch/Shader/Common.hlslh
    Source/Torch/Shader/FindAvailableNeighborsCs.hlsl
    Source/Torch/PyTorchBindings.cpp
    Source/Torch/SubMConv.cpp
    Source/Torch/SubMConv.hpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_subm_conv_files})

add_library(AIHoloImagerSubMConv SHARED
    ${aihi_subm_conv_files}
)

AddShaderFile(Source/Torch/Shader/BuildCoordHashCs.hlsl "cs" "main")
AddShaderFile(Source/Torch/Shader/FindAvailableNeighborsCs.hlsl "cs" "main")

set(torch_dir ${PROJECT_SOURCE_DIR}/External/PythonVenv/PythonVenv/Lib/site-packages/torch)

target_include_directories(AIHoloImagerSubMConv
    PRIVATE
        ${torch_dir}/include
        ${torch_dir}/include/torch/csrc/api/include
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

target_precompile_headers(AIHoloImagerSubMConv
    PRIVATE
        Source/pch.hpp
)

if(MSVC)
    target_compile_options(AIHoloImagerSubMConv
        PRIVATE
            /wd4702 # Ignore unreachable code in PyTorch
    )
endif()

target_compile_definitions(AIHoloImagerSubMConv
    PRIVATE
        TORCH_API_INCLUDE_EXTENSION_H
)

target_link_directories(AIHoloImagerSubMConv
    PRIVATE
        ${torch_dir}/lib
)

target_link_libraries(AIHoloImagerSubMConv
    PRIVATE
        AIHoloImagerGpuSystem
        AIHoloImagerTensorConverter

        c10.lib
        torch_cpu.lib
        torch.lib
        torch_python.lib
        Python3::Python
        glm
)

set_target_properties(AIHoloImagerSubMConv
    PROPERTIES
        SUFFIX ".pyd"
        POSTFIX ""
        DEBUG_POSTFIX ""
        MINSIZEREL_POSTFIX ""
        RELWITHDEBINFO_POSTFIX ""
        RELEASE_POSTFIX ""
)

add_dependencies(AIHoloImagerSubMConv DeployPython)
