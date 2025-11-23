cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

set(aihi_cmake_module_dir "${CMAKE_CURRENT_SOURCE_DIR}/CMake/Modules")
list(APPEND CMAKE_MODULE_PATH "${aihi_cmake_module_dir}")

set(ai_holo_imager_major_version 0)
set(ai_holo_imager_minor_version 5)
set(ai_holo_imager_patch_version 0)
set(ai_holo_imager_version ${ai_holo_imager_major_version}.${ai_holo_imager_minor_version}.${ai_holo_imager_patch_version})

project(AIHoloImager LANGUAGES CXX VERSION ${ai_holo_imager_version})

if(DEFINED ENV{BUILD_BUILDID})
    set(is_ci_env TRUE)
else()
    set(is_ci_env FALSE)
endif()

include(${aihi_cmake_module_dir}/Platform.cmake)
include(${aihi_cmake_module_dir}/Compiler.cmake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    set(aihi_output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR})
else()
    set(aihi_output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

option(AIHI_KEEP_INTERMEDIATES "Keep intermediate results" OFF)
option(AIHI_ENABLE_D3D12 "Enable D3D12" ON)

# 3.13.4 has issues with lib name. See https://github.com/python/cpython/issues/135151
find_package(Python3 3.11...<3.13.4 COMPONENTS Interpreter Development.Embed REQUIRED)

add_subdirectory(External)
add_subdirectory(Source)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT "AIHoloImager")
