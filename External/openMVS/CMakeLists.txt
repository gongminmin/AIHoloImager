# OpenMVS is in AGPL license. We need to use its binaries as-is, and communicate with it by data files.

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64.7z)
    message(STATUS "Downloading OpenMVS_Windows_x64.7z...")
    file(DOWNLOAD "https://github.com/cdcseacave/openMVS/releases/download/v2.3.0/OpenMVS_Windows_x64.7z" ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64.7z SHOW_PROGRESS)
endif()

if((NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64) OR (NOT IS_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64))
    message(STATUS "Extracting OpenMVS_Windows_x64...")
    file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64.7z DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64)
endif()

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR})
else()
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

set(deploy_files
    bz2.dll
    boost_iostreams-vc143-mt-x64-1_83.dll
    boost_program_options-vc143-mt-x64-1_83.dll
    boost_serialization-vc143-mt-x64-1_83.dll    
    DensifyPointCloud.exe
    gmp-10.dll
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
    ReconstructMesh.exe
    RefineMesh.exe
    TextureMesh.exe
    tiff.dll
    zstd.dll
)
set(depends_files)
foreach(file ${deploy_files})
    add_custom_command(OUTPUT ${output_dir}/${file}
        COMMAND ${CMAKE_COMMAND} -E create_hardlink
            ${CMAKE_CURRENT_BINARY_DIR}/OpenMVS_Windows_x64/${file}
            ${output_dir}/${file}
        COMMENT "Deploying ${file}..."
    )
    list(APPEND depends_files ${output_dir}/${file})
endforeach()

add_custom_target(DeployOpenMVS ALL
    DEPENDS ${depends_files}
    WORKING_DIRECTORY ${output_dir}
)

set_target_properties(DeployOpenMVS PROPERTIES FOLDER "External")
