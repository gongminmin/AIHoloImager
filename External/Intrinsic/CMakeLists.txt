# Copyright (c) 2025 Minmin Gong
#

if(is_ci_env)
    set(copy_files )
else()
    set(copy_files
        stage_0.pt
        stage_1.pt
        stage_2.pt
        stage_3.pt
    )
endif()

if(NOT is_ci_env)
    set(url "https://github.com/compphoto/Intrinsic/releases/download/v2.0/")
    set(local_dir "${CMAKE_CURRENT_SOURCE_DIR}/IntrinsicModels/")
    foreach(src_file ${copy_files})
        DownloadFile("${url}${src_file}" "${local_dir}${src_file}")
    endforeach()
endif()

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

set_target_properties(DeployIntrinsicModels PROPERTIES FOLDER "External")
