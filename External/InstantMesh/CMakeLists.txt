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
set(pyc_files )
foreach(py_file ${py_files})
    get_filename_component(py_file_dir ${py_file} DIRECTORY)
    get_filename_component(py_file_name ${py_file} NAME)
    get_filename_component(py_file_stem ${py_file} NAME_WE)
    set(py_file_dst "${output_dir}/${py_file_dir}/${py_file_stem}.pyc")
    add_custom_command(OUTPUT ${py_file_dst}
        COMMAND ${Python3_EXECUTABLE} -m compileall -b ${py_file}
        COMMAND ${CMAKE_COMMAND} -E copy ${py_file_dir}/${py_file_stem}.pyc ${py_file_dst}
        COMMENT "Compiling ${py_file_name} ..."
        DEPENDS ${py_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )

    list(APPEND pyc_files ${py_file_dst})
endforeach()

# zero123plus files need to be in source form
set(py_files
    InstantMesh/zero123plus/model.py
    InstantMesh/zero123plus/pipeline.py
)
foreach(py_file ${py_files})
    get_filename_component(py_file_dir ${py_file} DIRECTORY)
    get_filename_component(py_file_name ${py_file} NAME)
    set(py_file_dst "${output_dir}/${py_file_dir}/${py_file_name}")
    add_custom_command(OUTPUT ${py_file_dst}
        COMMAND ${CMAKE_COMMAND} -E copy ${py_file_dir}/${py_file_name} ${py_file_dst}
        COMMENT "Copying ${py_file_name} ..."
        DEPENDS ${py_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )

    list(APPEND pyc_files ${py_file_dst})
endforeach()

add_custom_target(DeployInstantMesh DEPENDS ${pyc_files})

set_target_properties(DeployInstantMesh PROPERTIES FOLDER "External")
