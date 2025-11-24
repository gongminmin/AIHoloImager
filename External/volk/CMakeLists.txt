# Copyright (c) 2025 Minmin Gong
#

UpdateExternalLib("volk" "https://github.com/zeux/volk.git" "59660878571aa99e3c9a366bb1d19fdcd701f0e7")

set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Vulkan-Headers/Vulkan-Headers" CACHE PATH "" FORCE)

if(WIN32)
   set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
endif()

add_subdirectory(volk EXCLUDE_FROM_ALL)

set_target_properties(volk PROPERTIES FOLDER "External")
