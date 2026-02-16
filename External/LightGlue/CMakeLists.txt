# Copyright (c) 2026 Minmin Gong
#

if(is_ci_env)
    add_custom_target(DeployLightGlueModels)
else()
    set(lg_version "v0.1_arxiv")
    set(url "https://github.com/cvg/LightGlue/releases/download/${lg_version}/")
    set(local_dir "${CMAKE_CURRENT_SOURCE_DIR}/LightGlueModels/")

    set(deployed_files )
    foreach(file "superpoint_v1" "superpoint_lightglue")
        set(src_file "${file}.pth")
        DownloadFile("${url}${src_file}" "${local_dir}${src_file}")

        set(output_file ${aihi_output_dir}/Models/LightGlue/${src_file})
        set(src_file LightGlueModels/${src_file})
        add_custom_command(OUTPUT ${output_file}
            COMMAND ${CMAKE_COMMAND} -E create_hardlink ${src_file} ${output_file}
            COMMENT "Deploying ${src_file}..."
            DEPENDS ${src_file}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            VERBATIM COMMAND_EXPAND_LISTS
        )

        list(APPEND deployed_files ${output_file})
    endforeach()

    add_custom_target(DeployLightGlueModels DEPENDS ${deployed_files})
endif()

set_target_properties(DeployLightGlueModels PROPERTIES FOLDER "External")
