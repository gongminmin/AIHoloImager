cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(ai_holo_imager_cmake_module_dir "${CMAKE_CURRENT_SOURCE_DIR}/CMake/Modules")
list(APPEND CMAKE_MODULE_PATH "${ai_holo_imager_cmake_module_dir}")

set(ai_holo_imager_major_version 0)
set(ai_holo_imager_minor_version 1)
set(ai_holo_imager_patch_version 0)
set(ai_holo_imager_version ${ai_holo_imager_major_version}.${ai_holo_imager_minor_version}.${ai_holo_imager_patch_version})

project(AIHoloImager LANGUAGES CXX VERSION ${ai_holo_imager_version})

include(${ai_holo_imager_cmake_module_dir}/Common.cmake)
include(${ai_holo_imager_cmake_module_dir}/Platform.cmake)
include(${ai_holo_imager_cmake_module_dir}/Compiler.cmake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

add_subdirectory(External)
add_subdirectory(Source)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT "AIHoloImager")
