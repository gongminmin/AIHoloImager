# Copyright (c) 2025 Minmin Gong
#

set(aihi_tensor_converter_files
    Include/TensorConverter/TensorConverter.hpp
    Source/MiniCudaRt.cpp
    Source/MiniCudaRt.hpp
    Source/TensorConverter.cpp
    Source/pch.hpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_tensor_converter_files})

add_library(AIHoloImagerTensorConverter SHARED
    ${aihi_tensor_converter_files}
)

set(torch_dir ${PROJECT_SOURCE_DIR}/External/PythonVenv/PythonVenv/Lib/site-packages/torch)

target_include_directories(AIHoloImagerTensorConverter
    PUBLIC
        Include
        ${torch_dir}/include
        ${torch_dir}/include/torch/csrc/api/include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

target_precompile_headers(AIHoloImagerTensorConverter
    PRIVATE
        Source/pch.hpp
)

if(MSVC)
    target_compile_options(AIHoloImagerTensorConverter
        PRIVATE
            /wd4702 # Ignore unreachable code in PyTorch
    )
endif()

target_link_directories(AIHoloImagerTensorConverter
    PRIVATE
        ${torch_dir}/lib
)

target_link_libraries(AIHoloImagerTensorConverter
    PUBLIC
        AIHoloImagerBase
        AIHoloImagerGpuSystem

    PRIVATE
        c10.lib
        torch_cpu.lib
        torch.lib
)
