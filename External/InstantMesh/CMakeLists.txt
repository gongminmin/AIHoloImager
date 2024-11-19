UpdateExternalLib("InstantMesh" "https://github.com/TencentARC/InstantMesh.git" "7fe95627cf819748f7830b2b278f302a9d798d17")

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR})
else()
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

set(py_files
    InstantMesh/src/__init__.py
    InstantMesh/src/models/__init__.py
    InstantMesh/src/models/decoder/__init__.py
    InstantMesh/src/models/decoder/transformer.py
    InstantMesh/src/models/encoder/__init__.py
    InstantMesh/src/models/encoder/dino.py
    InstantMesh/src/models/encoder/dino_wrapper.py
    InstantMesh/src/utils/__init__.py
    InstantMesh/src/utils/camera_util.py
)
foreach(src_file ${py_files})
    get_filename_component(src_file_dir ${src_file} DIRECTORY)
    get_filename_component(src_file_stem ${src_file} NAME_WE)
    set(pyc_file "${src_file_dir}/${src_file_stem}.pyc")
    set(dst_file "${output_dir}/${pyc_file}")
    add_custom_command(OUTPUT ${dst_file}
        COMMAND ${Python3_EXECUTABLE} -m compileall -b ${src_file}
        COMMAND ${CMAKE_COMMAND} -E copy ${pyc_file} ${dst_file}
        COMMENT "Compiling ${src_file} ..."
        MAIN_DEPENDENCY ${src_file}
        DEPENDS ${src_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )
endforeach()

# zero123plus files need to be in source form
set(copy_files
    InstantMesh/zero123plus/model.py
    InstantMesh/zero123plus/pipeline.py
)
foreach(src_file ${copy_files})
    set(dst_file "${output_dir}/${src_file}")
    add_custom_command(OUTPUT ${dst_file}
        COMMAND ${CMAKE_COMMAND} -E copy ${src_file} ${dst_file}
        COMMENT "Copying ${src_file} ..."
        MAIN_DEPENDENCY ${src_file}
        DEPENDS ${src_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )
endforeach()

set(instant_mesh_files
    ${py_files}
    ${copy_files}
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/InstantMesh FILES ${instant_mesh_files})

add_library(InstantMesh INTERFACE
    ${instant_mesh_files}
)

set_target_properties(InstantMesh PROPERTIES FOLDER "External")
