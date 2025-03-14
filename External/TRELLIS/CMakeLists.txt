# Copyright (c) 2025 Minmin Gong
#

if(DEFINED ENV{GIT_LFS_SKIP_SMUDGE})
    set(lfs_skip "$ENV{GIT_LFS_SKIP_SMUDGE}")
else()
    set(lfs_skip 0)
endif()

set(ENV{GIT_LFS_SKIP_SMUDGE} 1)
UpdateExternalLib("TRELLIS-image-large" "https://huggingface.co/JeffreyXiang/TRELLIS-image-large" "25e0d31ffbebe4b5a97464dd851910efc3002d96" "" "" need_checkout)
set(ENV{GIT_LFS_SKIP_SMUDGE} ${lfs_skip})

if((NOT is_ci_env) AND need_checkout) # CI server doesn't have enough disk space for them
    PullLfsFile("TRELLIS-image-large" "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors")
    PullLfsFile("TRELLIS-image-large" "ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors")
    PullLfsFile("TRELLIS-image-large" "ckpts/ss_dec_conv3d_16l8_fp16.safetensors")
    PullLfsFile("TRELLIS-image-large" "ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors")
endif()

set(deployed_files )

set(copy_files
    ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json
    ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors
    ckpts/slat_flow_img_dit_L_64l8p2_fp16.json
    ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors
    ckpts/ss_dec_conv3d_16l8_fp16.json
    ckpts/ss_dec_conv3d_16l8_fp16.safetensors
    ckpts/ss_flow_img_dit_L_16l8_fp16.json
    ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors
    pipeline.json
)
foreach(src_file ${copy_files})
    set(src_file TRELLIS-image-large/${src_file})
    set(output_file ${aihi_output_dir}/Models/${src_file})
    add_custom_command(OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND} -E create_hardlink ${src_file} ${output_file}
        COMMENT "Deploying ${src_file}..."
        DEPENDS ${src_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )

    list(APPEND deployed_files ${output_file})
endforeach()

add_custom_target(DeployTrellisModels DEPENDS ${deployed_files})
add_dependencies(DeployTrellisModels DeployDinov2Models)

set_target_properties(DeployTrellisModels PROPERTIES FOLDER "External")
