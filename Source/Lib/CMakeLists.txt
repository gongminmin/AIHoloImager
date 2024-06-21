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

add_library(AIHoloImagerLib STATIC
    Include/AIHoloImager/AIHoloImager.hpp
    Include/AIHoloImager/Mesh.hpp
    Include/AIHoloImager/Texture.hpp
    Source/AIHoloImager.cpp
    Source/Mesh.cpp
    Source/Texture.cpp
    ${sfm_source_files}
    ${sfm_header_files}
    ${camera_database_src}
)

target_include_directories(AIHoloImagerLib
    PUBLIC
        Include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

target_link_libraries(AIHoloImagerLib
    PRIVATE
        assimp
        openMVG_exif
        openMVG_matching_image_collection
        openMVG_sfm
        stb
)
