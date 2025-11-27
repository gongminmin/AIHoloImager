# Copyright (c) 2024-2025 Minmin Gong
#

UpdateExternalLib(NAME "openMVG"
                  URL "https://github.com/openMVG/openMVG.git"
                  REV "c92ed1be7c25068c0cdc0927e876a9ec92e97ce8"
                  RECURSIVE_SUBMODULES
                  NEED_CHECKOUT need_patch
)
if(need_patch)
    foreach(patch "0001-Use-PROJECT_SOURCE_DIR-to-replace-CMAKE_SOURCE_DIR" "0002-Enable-to-load-from-memory-data-in-Matches_Provider-" "0003-Remove-pragma-warning-once-so-that-these-warnings-ca")
        ApplyPatch("openMVG" "${CMAKE_CURRENT_SOURCE_DIR}/Patches/${patch}.patch")
    endforeach()
endif()

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(OpenMVG_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(OpenMVG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OpenMVG_BUILD_GUI_SOFTWARES OFF CACHE BOOL "" FORCE)
set(OpenMVG_BUILD_OPENGL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OpenMVG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(OpenMVG_BUILD_SOFTWARES OFF CACHE BOOL "" FORCE)
set(OpenMVG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OpenMVG_USE_LIGT OFF CACHE BOOL "" FORCE)
set(OPENMVG_USE_OPENMP ON CACHE BOOL "" FORCE)
set(OpenMVG_USE_RERUN OFF CACHE BOOL "" FORCE)

# Force to use OpenMVG's internal dependencies
set(CMAKE_DISABLE_FIND_PACKAGE_cereal TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_Eigen3 TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_Ceres TRUE)

add_subdirectory(openMVG/src EXCLUDE_FROM_ALL)

list(REMOVE_ITEM CMAKE_CONFIGURATION_TYPES "Maintainer")
set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "" FORCE)

get_all_targets(all_targets)
foreach(target ${all_targets})
    get_target_property(vs_folder ${target} FOLDER)
    if(NOT vs_folder)
        set(vs_folder "")
    endif()
    set_target_properties(${target} PROPERTIES FOLDER "External/${vs_folder}")

    if(ai_holo_imager_compiler_msvc OR ai_holo_imager_compiler_clangcl)
        set(remove_flags /Wall /W4 /W3 /W2 /W1)
        set(add_flags /W0 /wd4244 /wd4267)
    else()
        set(remove_flags -Wall -Wextra)
        set(add_flags -w)
    endif()

    foreach(flag_var COMPILE_FLAGS COMPILE_OPTIONS)
        get_target_property(compile_opts ${target} ${flag_var})
        if(compile_opts)
            foreach(warning_flag ${remove_flags})
                string(REPLACE "${warning_flag}" "" ${compile_opts} "${${compile_opts}}")
            endforeach()
        endif()
    endforeach()

    get_target_property(target_type ${target} TYPE)
    if ((target_type MATCHES "LIBRARY") AND (NOT target_type STREQUAL "INTERFACE_LIBRARY"))
        target_compile_options(${target} PRIVATE ${add_flags})
    endif()
endforeach()
