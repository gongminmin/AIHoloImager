# Copyright (c) 2025 Minmin Gong
#

UpdateExternalLib("SPIRV-Reflect" "https://github.com/KhronosGroup/SPIRV-Reflect.git" "ef913b3ab3da1becca3cf46b15a10667c67bebe5")

set(SPIRV_REFLECT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_ENABLE_ASAN OFF CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_INSTALL OFF CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "" FORCE)

add_subdirectory(SPIRV-Reflect EXCLUDE_FROM_ALL)

set_target_properties(spirv-reflect-static PROPERTIES FOLDER "External")
