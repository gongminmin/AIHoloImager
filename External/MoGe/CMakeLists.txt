# Copyright (c) 2025 Minmin Gong
#

if(is_ci_env)
    add_custom_target(DeployMoGeModels)
else()
    if(DEFINED ENV{GIT_LFS_SKIP_SMUDGE})
        set(lfs_skip "$ENV{GIT_LFS_SKIP_SMUDGE}")
    else()
        set(lfs_skip 0)
    endif()

    set(ENV{GIT_LFS_SKIP_SMUDGE} 1)
    UpdateExternalLib("moge-vitl" "https://huggingface.co/Ruicheng/moge-vitl" "979e84da9415762c30e6c0cf8dc0962896c793df" "" "" need_checkout)
    set(ENV{GIT_LFS_SKIP_SMUDGE} ${lfs_skip})

    if(need_checkout)
        PullLfsFile("moge-vitl" "model.pt")
    endif()

    set(deployed_files )

    set(copy_files
        model.pt
    )
    foreach(src_file ${copy_files})
        set(src_file moge-vitl/${src_file})
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

    add_custom_target(DeployMoGeModels DEPENDS ${output_file})
endif()

set_target_properties(DeployMoGeModels PROPERTIES FOLDER "External")
