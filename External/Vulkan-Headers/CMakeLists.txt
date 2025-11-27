# Copyright (c) 2025 Minmin Gong
#

UpdateExternalLib(NAME "Vulkan-Headers"
                  URL "https://github.com/KhronosGroup/Vulkan-Headers.git"
                  REV "6aefb8eb95c8e170d0805fd0f2d02832ec1e099a"
)

set(VULKAN_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(ULKAN_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_MODULE OFF CACHE BOOL "" FORCE)

add_subdirectory(Vulkan-Headers EXCLUDE_FROM_ALL)

set_target_properties(Vulkan-Headers PROPERTIES FOLDER "External")
