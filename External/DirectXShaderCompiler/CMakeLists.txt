# Copyright (c) 2025-2026 Minmin Gong
#

set(prebuilt_dxc_url "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2505.1/dxc_2025_07_14.zip")
cmake_path(GET prebuilt_dxc_url FILENAME prebuilt_dxc_name)
cmake_path(GET prebuilt_dxc_name STEM prebuilt_dxc_stem)

set(prebuilt_dxc_tmp_dir "${CMAKE_CURRENT_SOURCE_DIR}/PrebuiltDxc")
set(prebuilt_dxc_path "${prebuilt_dxc_tmp_dir}/${prebuilt_dxc_name}")
if(NOT EXISTS ${prebuilt_dxc_path})
    message(STATUS "Downloading prebuilt dxc package from ${prebuilt_dxc_url}")
    file(DOWNLOAD ${prebuilt_dxc_url} ${prebuilt_dxc_path} SHOW_PROGRESS)
endif()

set(prebuilt_dxc_dir "${prebuilt_dxc_tmp_dir}/${prebuilt_dxc_stem}")
if (NOT EXISTS ${prebuilt_dxc_dir})
    message(STATUS "Extracting prebuilt dxc package to ${prebuilt_dxc_dir}")
    file(ARCHIVE_EXTRACT INPUT "${prebuilt_dxc_path}"
        DESTINATION "${prebuilt_dxc_dir}"
    )
endif()

set(aihi_dxc_cmd "${prebuilt_dxc_dir}/bin/x64/dxc.exe")
cmake_path(ABSOLUTE_PATH aihi_dxc_cmd OUTPUT_VARIABLE aihi_dxc_cmd)
set(AIHI_DXC_CMD ${aihi_dxc_cmd} CACHE INTERNAL "")

set(copy_files
    dxcompiler.dll
)
foreach(src_file ${copy_files})
    set(output_file ${aihi_output_dir}/${src_file})
    set(src_file ${prebuilt_dxc_dir}/bin/x64/${src_file})
    add_custom_command(OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND} -E create_hardlink ${src_file} ${output_file}
        COMMENT "Deploying ${src_file}..."
        DEPENDS ${src_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )

    list(APPEND deployed_files ${output_file})
endforeach()

add_custom_target(DeployDxc DEPENDS ${output_file})

set_target_properties(DeployDxc PROPERTIES FOLDER "External")
