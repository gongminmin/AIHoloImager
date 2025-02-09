# Copyright (c) 2025 Minmin Gong
#

UpdateExternalLib("MoGe" "https://github.com/microsoft/MoGe.git" "dd158c05461f2353287a182afb2adf0fda46436f")

set(moge_files
    MoGe/moge/model/dinov2/hub/backbones.py
    MoGe/moge/model/dinov2/hub/utils.py
    MoGe/moge/model/dinov2/hub/__init__.py
    MoGe/moge/model/dinov2/layers/attention.py
    MoGe/moge/model/dinov2/layers/block.py
    MoGe/moge/model/dinov2/layers/dino_head.py
    MoGe/moge/model/dinov2/layers/drop_path.py
    MoGe/moge/model/dinov2/layers/layer_scale.py
    MoGe/moge/model/dinov2/layers/mlp.py
    MoGe/moge/model/dinov2/layers/patch_embed.py
    MoGe/moge/model/dinov2/layers/swiglu_ffn.py
    MoGe/moge/model/dinov2/layers/__init__.py
    MoGe/moge/model/dinov2/models/vision_transformer.py
    MoGe/moge/model/dinov2/models/__init__.py
    MoGe/moge/model/dinov2/__init__.py
    MoGe/moge/model/moge_model.py
    MoGe/moge/model/utils.py
    MoGe/moge/model/__init__.py
    MoGe/moge/utils/geometry_numpy.py
    MoGe/moge/utils/geometry_torch.py
    MoGe/moge/utils/io.py
    MoGe/moge/utils/tools.py
    MoGe/moge/utils/vis.py
    MoGe/moge/utils/__init__.py
    MoGe/utils3d/numpy/mesh.py
    MoGe/utils3d/numpy/transforms.py
    MoGe/utils3d/numpy/utils.py
    MoGe/utils3d/numpy/_helpers.py
    MoGe/utils3d/numpy/__init__.py
    MoGe/utils3d/torch/mesh.py
    MoGe/utils3d/torch/transforms.py
    MoGe/utils3d/torch/utils.py
    MoGe/utils3d/torch/_helpers.py
    MoGe/utils3d/torch/__init__.py
    MoGe/utils3d/_unified/__init__.py
    MoGe/utils3d/_helpers.py
    MoGe/utils3d/__init__.py
)
foreach(src_file ${moge_files})
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

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/MoGe FILES ${moge_files})

add_library(MoGe INTERFACE
    ${moge_files}
)

set_target_properties(MoGe PROPERTIES FOLDER "External")
