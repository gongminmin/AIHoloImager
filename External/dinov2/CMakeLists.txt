UpdateExternalLib("dinov2" "https://github.com/facebookresearch/dinov2.git" "e1277af2ba9496fbadf7aec6eba56e8d882d1e35")

set(py_files
    dinov2/dinov2/hub/depth/decode_heads.py
    dinov2/dinov2/hub/depth/encoder_decoder.py
    dinov2/dinov2/hub/depth/ops.py
    dinov2/dinov2/hub/depth/__init__.py
    dinov2/dinov2/hub/backbones.py
    dinov2/dinov2/hub/classifiers.py
    dinov2/dinov2/hub/depthers.py
    dinov2/dinov2/hub/utils.py
    dinov2/dinov2/hub/__init__.py
    dinov2/dinov2/layers/attention.py
    dinov2/dinov2/layers/block.py
    dinov2/dinov2/layers/dino_head.py
    dinov2/dinov2/layers/drop_path.py
    dinov2/dinov2/layers/layer_scale.py
    dinov2/dinov2/layers/mlp.py
    dinov2/dinov2/layers/patch_embed.py
    dinov2/dinov2/layers/swiglu_ffn.py
    dinov2/dinov2/layers/__init__.py
    dinov2/dinov2/models/vision_transformer.py
    dinov2/dinov2/models/__init__.py
    dinov2/dinov2/__init__.py
)
foreach(src_file ${py_files})
    get_filename_component(src_file_dir ${src_file} DIRECTORY)
    get_filename_component(src_file_stem ${src_file} NAME_WE)
    set(pyc_file "${src_file_dir}/${src_file_stem}.pyc")
    set(dst_file "${aihi_output_dir}/${pyc_file}")
    add_custom_command(OUTPUT ${dst_file}
        COMMAND ${Python3_EXECUTABLE} -m compileall -b -o $<IF:$<CONFIG:Debug>,0,1> ${src_file}
        COMMAND ${CMAKE_COMMAND} -E copy ${pyc_file} ${dst_file}
        COMMENT "Compiling ${src_file} ..."
        MAIN_DEPENDENCY ${src_file}
        DEPENDS ${src_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )
endforeach()

set(copy_files
    dinov2/hubconf.py
)
foreach(src_file ${copy_files})
    set(dst_file "${aihi_output_dir}/${src_file}")
    add_custom_command(OUTPUT ${dst_file}
        COMMAND ${CMAKE_COMMAND} -E copy ${src_file} ${dst_file}
        COMMENT "Copying ${src_file} ..."
        MAIN_DEPENDENCY ${src_file}
        DEPENDS ${src_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )
endforeach()

set(dinov2_files
    ${py_files}
    ${copy_files}
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${dinov2_files})

add_library(dinov2 INTERFACE
    ${dinov2_files}
)

set_target_properties(dinov2 PROPERTIES FOLDER "External")
