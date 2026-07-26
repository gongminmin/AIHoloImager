# Copyright (c) 2025-2026 Minmin Gong
#

if(is_ci_env)
    add_custom_target(DeployIntrinsicModels)
else()
    set(copy_files
        stage_0.pt
        stage_1.pt
        stage_2.pt
        stage_3.pt
    )

    set(url "https://github.com/compphoto/Intrinsic/releases/download/v2.0/")
    set(local_dir "${CMAKE_CURRENT_SOURCE_DIR}/IntrinsicModels/")
    foreach(src_file ${copy_files})
        DownloadFile("${url}${src_file}" "${local_dir}${src_file}")
    endforeach()

    set(deployed_files )
    foreach(src_file ${copy_files})
        set(output_file ${aihi_output_dir}/Models/Intrinsic/${src_file})
        set(src_file IntrinsicModels/${src_file})
        add_custom_command(OUTPUT ${output_file}
            COMMAND ${CMAKE_COMMAND} -E create_hardlink ${src_file} ${output_file}
            COMMENT "Deploying ${src_file}..."
            DEPENDS ${src_file}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            VERBATIM COMMAND_EXPAND_LISTS
        )

        list(APPEND deployed_files ${output_file})
    endforeach()

    add_custom_target(DeployIntrinsicModels DEPENDS ${deployed_files})
endif()

set_target_properties(DeployIntrinsicModels PROPERTIES FOLDER "External")
