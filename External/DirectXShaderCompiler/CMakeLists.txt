# Copyright (c) 2025 Minmin Gong
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
