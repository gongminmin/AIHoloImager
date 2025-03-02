# OpenMVS is in AGPL license. We need to use its binaries as-is, and communicate with it by data files.

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64.7z)
    message(STATUS "Downloading OpenMVS_Windows_x64.7z...")
    file(DOWNLOAD "https://github.com/cdcseacave/openMVS/releases/download/v2.3.0/OpenMVS_Windows_x64.7z" ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64.7z SHOW_PROGRESS)
endif()

if((NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64) OR (NOT IS_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64))
    message(STATUS "Extracting OpenMVS_Windows_x64...")
    file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64.7z DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64)
endif()

set(deployed_files )

set(copy_files
    bz2.dll
    boost_iostreams-vc143-mt-x64-1_83.dll
    boost_program_options-vc143-mt-x64-1_83.dll
    boost_serialization-vc143-mt-x64-1_83.dll
    DensifyPointCloud.exe
    Iex-3_1.dll
    IlmThread-3_1.dll
    Imath-3_1.dll
    jpeg62.dll
    liblzma.dll
    libpng16.dll
    libsharpyuv.dll
    libwebp.dll
    libwebpdecoder.dll
    opencv_calib3d4.dll
    opencv_core4.dll
    opencv_features2d4.dll
    opencv_flann4.dll
    opencv_imgcodecs4.dll
    opencv_imgproc4.dll
    OpenEXR-3_1.dll
    tiff.dll
    zstd.dll
)
foreach(src_file ${copy_files})
    add_custom_command(OUTPUT ${aihi_output_dir}/${src_file}
        COMMAND ${CMAKE_COMMAND} -E create_hardlink OpenMVS_Windows_x64/${src_file} ${aihi_output_dir}/${src_file}
        COMMENT "Deploying ${src_file}..."
        DEPENDS OpenMVS_Windows_x64/${src_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )

    list(APPEND deployed_files ${aihi_output_dir}/${src_file})
endforeach()

add_custom_target(DeployOpenMVS DEPENDS ${deployed_files})

set_target_properties(DeployOpenMVS PROPERTIES FOLDER "External")
