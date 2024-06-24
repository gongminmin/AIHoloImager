set(camera_database_src "${PROJECT_SOURCE_DIR}/External/openMVG/openMVG/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt")
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR})
else()
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()
set(camera_database_dst "${output_dir}/sensor_width_camera_database.txt")
add_custom_command(OUTPUT ${camera_database_dst}
    COMMAND ${CMAKE_COMMAND} -E copy ${camera_database_src} ${camera_database_dst}
    COMMENT "Copying camera database ..."
    MAIN_DEPENDENCY ${camera_database_src}
    DEPENDS ${camera_database_src}
    VERBATIM COMMAND_EXPAND_LISTS
)

set(sfm_source_files
    Source/SfM/StructureFromMotion.cpp
)
set(sfm_header_files
    Source/SfM/StructureFromMotion.hpp
)
source_group("Source Files\\SfM" FILES ${sfm_source_files})
source_group("Header Files\\SfM" FILES ${sfm_header_files})

set(mesh_recon_source_files
    Source/MeshRecon/MeshReconstruction.cpp
)
set(mesh_recon_header_files
    Source/MeshRecon/MeshReconstruction.hpp
)
source_group("Source Files\\MeshRecon" FILES ${mesh_recon_source_files})
source_group("Header Files\\MeshRecon" FILES ${mesh_recon_header_files})

set(mask_gen_source_files
    Source/MaskGen/MaskGenerator.cpp
    Source/MaskGen/MaskGenerator.py
)
set(mask_gen_header_files
    Source/MaskGen/MaskGenerator.hpp
)
source_group("Source Files\\MaskGen" FILES ${mask_gen_source_files})
source_group("Header Files\\MaskGen" FILES ${mask_gen_header_files})

add_library(AIHoloImagerLib STATIC
    Include/AIHoloImager/AIHoloImager.hpp
    Include/AIHoloImager/Mesh.hpp
    Include/AIHoloImager/Texture.hpp
    Source/AIHoloImager.cpp
    Source/Mesh.cpp
    Source/Texture.cpp
    ${camera_database_src}
    ${sfm_source_files}
    ${sfm_header_files}
    ${mesh_recon_source_files}
    ${mesh_recon_header_files}
    ${mask_gen_source_files}
    ${mask_gen_header_files}
)

target_include_directories(AIHoloImagerLib
    PUBLIC
        Include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/External/openMVG/openMVG/src/software/SfM/export
)

find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)
string(REPLACE "\\" "/" AIHI_PY_STDLIB_DIR ${Python3_STDLIB})
string(REPLACE "\\" "/" AIHI_PY_RUNTIME_LIB_DIR ${Python3_RUNTIME_LIBRARY_DIRS})

target_compile_definitions(AIHoloImagerLib
    PRIVATE
        AIHI_PY_STDLIB_DIR="${AIHI_PY_STDLIB_DIR}"
        AIHI_PY_RUNTIME_LIB_DIR="${AIHI_PY_RUNTIME_LIB_DIR}"
)

target_link_libraries(AIHoloImagerLib
    PRIVATE
        assimp
        openMVG_exif
        openMVG_matching_image_collection
        openMVG_sfm
        Python3::Python
        stb
)

set(mask_generator_src "${CMAKE_CURRENT_SOURCE_DIR}/Source/MaskGen/MaskGenerator.py")
get_filename_component(mask_generator_dir ${mask_generator_src} DIRECTORY)
get_filename_component(mask_generator_file_name ${mask_generator_src} NAME)
get_filename_component(mask_generator_stem ${mask_generator_src} NAME_WE)
set(mask_generator_dst "${output_dir}/${mask_generator_stem}.pyc")
add_custom_command(OUTPUT ${mask_generator_dst}
    COMMAND ${Python3_EXECUTABLE} -m compileall -b ${mask_generator_src}
    COMMAND ${CMAKE_COMMAND} -E copy ${mask_generator_dir}/${mask_generator_stem}.pyc ${mask_generator_dst}
    COMMENT "Compiling ${mask_generator_file_name} ..."
    MAIN_DEPENDENCY ${mask_generator_src}
    DEPENDS ${mask_generator_src}
    VERBATIM COMMAND_EXPAND_LISTS
)

add_dependencies(AIHoloImagerLib DeployOpenMVS DeployPython)
