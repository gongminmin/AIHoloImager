# Copyright (c) 2024-2025 Minmin Gong
#

if(NOT EXISTS ${GIT_EXECUTABLE})
    unset(GIT_EXECUTABLE CACHE)
endif()

find_package(Git)
if(Git_FOUND)
    set(GIT_EXECUTABLE ${GIT_EXECUTABLE} CACHE INTERNAL "" FORCE)
else()
    message(FATAL_ERROR "Failed to find git.")
endif()
if(NOT EXISTS ${GIT_EXECUTABLE})
    message(FATAL_ERROR "The git doesn't exist.")
endif()

# CloneExternalLib
#
# Clone an external git repository.
#
# Parameters:
#   NAME            - Required: unique identifier for this external lib
#   URL             - Required: git repository URL
#   BRANCH          - Optional: if set, performs a shallow clone of this branch
#   SHALLOW_EXCLUDE - Optional: single revision to exclude from shallow history
#
function(CloneExternalLib)
    set(option_args )
    set(one_value_keyword_args NAME URL BRANCH SHALLOW_EXCLUDE)
    set(multi_value_keyword_args )

    cmake_parse_arguments(
        PARSE_ARGV 0
        ARG_PREFIX
        "${option_args}"
        "${one_value_keyword_args}"
        "${multi_value_keyword_args}"
    )

    if((NOT DEFINED ARG_PREFIX_NAME) OR (NOT DEFINED ARG_PREFIX_URL))
        message(FATAL_ERROR "NAME and URL are required")
    endif()

    set(name ${ARG_PREFIX_NAME})
    set(url ${ARG_PREFIX_URL})

    set(external_folder "${CMAKE_CURRENT_SOURCE_DIR}")
    set(external_lib_folder "${external_folder}/${name}")

    if(EXISTS ${external_lib_folder})
        message(STATUS "Removing ${external_lib_folder}...")
        file(REMOVE_RECURSE ${external_lib_folder})
    endif()

    message(STATUS "Cloning ${name}...")
    set(branch_param "")
    set(branch_name_param "")
    if(DEFINED ARG_PREFIX_BRANCH)
        set(branch_param "-b")
        set(branch_name_param "${ARG_PREFIX_BRANCH}")
    endif()
    set(shallow_exclude_param "")
    if(DEFINED ARG_PREFIX_SHALLOW_EXCLUDE)
        set(shallow_exclude_param "--shallow-exclude=${ARG_PREFIX_SHALLOW_EXCLUDE}")
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" clone ${url} ${branch_param} ${branch_name_param} ${shallow_exclude_param} "${name}" -n WORKING_DIRECTORY "${external_folder}")
endfunction()

# CheckoutExternalLib
#
# Fetches an external git repository.
#
# Parameters:
#   NAME                 - Required: unique identifier for this external lib
#   REV                  - Required: commit hash, tag, or branch to check out
#   RECURSIVE_SUBMODULES - Optional: ON/OFF boolean - if ON, recuisive submodule init/update
#   CHECKOUT_ERR         - Optional: output boolean variable indicating if a checkout was performed
#
function(CheckoutExternalLib)
    set(option_args RECURSIVE_SUBMODULES)
    set(one_value_keyword_args NAME REV CHECKOUT_ERR)
    set(multi_value_keyword_args )

    cmake_parse_arguments(
        PARSE_ARGV 0
        ARG_PREFIX
        "${option_args}"
        "${one_value_keyword_args}"
        "${multi_value_keyword_args}"
    )

    if((NOT DEFINED ARG_PREFIX_NAME) OR (NOT DEFINED ARG_PREFIX_REV))
        message(FATAL_ERROR "NAME and REV are required")
    endif()

    set(recursive_submodules OFF)
    if(DEFINED ARG_PREFIX_RECURSIVE_SUBMODULES)
        set(recursive_submodules ${ARG_PREFIX_RECURSIVE_SUBMODULES})
    endif()

    set(name ${ARG_PREFIX_NAME})
    set(rev ${ARG_PREFIX_REV})

    set(external_folder "${CMAKE_CURRENT_SOURCE_DIR}")
    set(external_lib_folder "${external_folder}/${name}")

    execute_process(COMMAND "${GIT_EXECUTABLE}" checkout -q ${rev} WORKING_DIRECTORY ${external_lib_folder} RESULT_VARIABLE checkout_err)
    if((NOT ${checkout_err}) AND ${recursive_submodules})
        execute_process(COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive WORKING_DIRECTORY ${external_lib_folder} RESULT_VARIABLE checkout_err)
    endif()

    if(DEFINED ARG_PREFIX_CHECKOUT_ERR)
        set(${ARG_PREFIX_NEED_CHECKOUT} ${checkout_err} PARENT_SCOPE)
    endif()
endfunction()

# UpdateExternalLib
#
# Update an external library with fine-grained control.
#
# Parameters:
#   NAME                 - Required: unique identifier for this external lib
#   URL                  - Required: git repository URL
#   REV                  - Required: commit hash, tag, or branch to check out
#   BRANCH               - Optional: if set, performs a shallow clone of this branch
#   SHALLOW_EXCLUDE      - Optional: single revision to exclude from shallow history
#   RECURSIVE_SUBMODULES - Optional: ON/OFF boolean - if ON, recuisive submodule init/update
#   NEED_CHECKOUT        - Optional: output boolean variable indicating if a checkout was performed
#
function(UpdateExternalLib)
    set(option_args RECURSIVE_SUBMODULES)
    set(one_value_keyword_args NAME URL REV BRANCH SHALLOW_EXCLUDE NEED_CHECKOUT)
    set(multi_value_keyword_args )

    cmake_parse_arguments(
        PARSE_ARGV 0
        ARG_PREFIX
        "${option_args}"
        "${one_value_keyword_args}"
        "${multi_value_keyword_args}"
    )

    if((NOT DEFINED ARG_PREFIX_NAME) OR (NOT DEFINED ARG_PREFIX_URL) OR (NOT DEFINED ARG_PREFIX_REV))
        message(FATAL_ERROR "NAME, URL and REV are required")
    endif()

    set(name ${ARG_PREFIX_NAME})
    set(url ${ARG_PREFIX_URL})
    set(rev ${ARG_PREFIX_REV})

    set(clone_args )
    if((DEFINED ARG_PREFIX_BRANCH) AND (NOT ARG_PREFIX_BRANCH STREQUAL ""))
        list(APPEND clone_args BRANCH "${ARG_PREFIX_BRANCH}")
    endif()
    if((DEFINED ARG_PREFIX_SHALLOW_EXCLUDE) AND (NOT ARG_PREFIX_SHALLOW_EXCLUDE STREQUAL ""))
        list(APPEND clone_args SHALLOW_EXCLUDE "${ARG_PREFIX_SHALLOW_EXCLUDE}")
    endif()

    set(checkout_args )
    if((DEFINED ARG_PREFIX_RECURSIVE_SUBMODULES) AND ARG_PREFIX_RECURSIVE_SUBMODULES)
        list(APPEND checkout_args "RECURSIVE_SUBMODULES")
    endif()

    set(external_folder "${CMAKE_CURRENT_SOURCE_DIR}")
    set(external_lib_folder "${external_folder}/${name}")

    if(EXISTS "${external_lib_folder}/.git")
        set(need_clone FALSE)

        set(cached_rev_file_name "${CMAKE_CURRENT_BINARY_DIR}/AIHI_${name}_REV")
        if(EXISTS "${cached_rev_file_name}")
            file(STRINGS "${cached_rev_file_name}" cached_rev)
        else()
            set(cached_rev "")
        endif()
        IF("${cached_rev}" STREQUAL "${rev}")
            set(need_checkout FALSE)
        else()
            message(STATUS "Updating ${name} to revision ${rev}...")
            execute_process(COMMAND "${GIT_EXECUTABLE}" "fetch" "origin" WORKING_DIRECTORY ${external_lib_folder})
            execute_process(COMMAND "${GIT_EXECUTABLE}" "rev-parse" "HEAD" WORKING_DIRECTORY ${external_lib_folder} OUTPUT_VARIABLE head_rev)
            string(STRIP ${head_rev} head_rev)
            if (${head_rev} STREQUAL ${rev})
                set(need_checkout FALSE)
            else()
                set(need_checkout TRUE)
            endif()
            file(WRITE "${cached_rev_file_name}" ${rev})
        endif()
    else()
        set(need_clone TRUE)
    endif()

    if(need_clone)
        CloneExternalLib(NAME ${name}
                         URL ${url}
                         ${clone_args}
        )
        set(need_checkout TRUE)
    endif()

    if(need_checkout)
        message(STATUS "Checking out to revision ${rev}...")
        CheckoutExternalLib(NAME ${name}
                            REV ${rev}
                            CHECKOUT_ERR checkout_err
                            ${checkout_args}
        )
        if(checkout_err)
            message(STATUS "COULD NOT checkout revision ${rev}, reclone the repository.")
            CloneExternalLib(NAME ${name}
                             URL ${url}
                             ${clone_args}
            )

            message(STATUS "Checking out to revision ${rev}...")
            CheckoutExternalLib(NAME ${name}
                                REV ${rev}
                                CHECKOUT_ERR checkout_err
                                ${checkout_args}
            )
        endif()
    endif()

    if(DEFINED ARG_PREFIX_NEED_CHECKOUT)
        set(${ARG_PREFIX_NEED_CHECKOUT} ${need_checkout} PARENT_SCOPE)
    endif()
endfunction()

function(ApplyPatch name patch)
    set(external_folder "${CMAKE_CURRENT_SOURCE_DIR}")
    set(external_lib_folder "${external_folder}/${name}")

    execute_process(COMMAND "${GIT_EXECUTABLE}" "apply" "--check" "--ignore-space-change" "${patch}" WORKING_DIRECTORY ${external_lib_folder} RESULT_VARIABLE checkout_err)
    if(checkout_err)
        message(FATAL_ERROR "Fail to apply patch ${patch}. It's very likely your git is not correctly configured.")
    else()
        message(STATUS "Applying ${patch}...")
        execute_process(COMMAND "${GIT_EXECUTABLE}" "am" "--ignore-space-change" "${patch}" WORKING_DIRECTORY ${external_lib_folder})
    endif()
endfunction()

function(PullLfsFile name path)
    set(external_lib_folder "${CMAKE_CURRENT_SOURCE_DIR}/${name}")
    message(STATUS "Pulling ${path}...")
    execute_process(COMMAND "${GIT_EXECUTABLE}" "lfs" "pull" "--include" "${path}" WORKING_DIRECTORY ${external_lib_folder} RESULT_VARIABLE pull_err)
    if(pull_err)
        message(FATAL_ERROR "COULD NOT pull the LFS file ${path}, error: ${pull_err}.")
    endif()
endfunction()

function(DownloadFile url local_path)
    if(NOT EXISTS ${local_path})
        get_filename_component(file_name ${local_path} NAME)
        message(STATUS "Downloading ${file_name}...")
        file(DOWNLOAD ${url} ${local_path} SHOW_PROGRESS)
    endif()
endfunction()

function(get_all_targets var)
    set(targets)
    get_all_targets_recursive(targets ${CMAKE_CURRENT_SOURCE_DIR})
    set(${var} ${targets} PARENT_SCOPE)
endfunction()

macro(get_all_targets_recursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        get_all_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()

add_subdirectory(zlib)

add_subdirectory(assimp)
add_subdirectory(cxxopts)
add_subdirectory(dinov2)
add_subdirectory(DirectX-Headers)
add_subdirectory(DirectXShaderCompiler)
add_subdirectory(glm)
add_subdirectory(Intrinsic)
add_subdirectory(MoGe)
add_subdirectory(openMVG)
add_subdirectory(PythonVenv)
add_subdirectory(SPIRV-Reflect)
add_subdirectory(stb)
add_subdirectory(TRELLIS)
add_subdirectory(U-2-Net)
add_subdirectory(volk)
add_subdirectory(Vulkan-Headers)
add_subdirectory(xatlas)
