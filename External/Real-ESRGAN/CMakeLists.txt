# Copyright (c) 2026 Minmin Gong
#

if(is_ci_env)
    add_custom_target(DeployRealESrganModels)
else()
    set(url "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/")
    set(local_dir "${CMAKE_CURRENT_SOURCE_DIR}/RealESrGanModel/")

    set(deployed_files )
    foreach(file "realesr-general-x4v3" "realesr-general-wdn-x4v3")
        set(src_file "${file}.pth")
        DownloadFile("${url}${src_file}" "${local_dir}${src_file}")

        set(output_file ${aihi_output_dir}/Models/Real-ESRGAN/${src_file})
        set(src_file RealESrGanModel/${src_file})
        add_custom_command(OUTPUT ${output_file}
            COMMAND ${CMAKE_COMMAND} -E create_hardlink ${src_file} ${output_file}
            COMMENT "Deploying ${src_file}..."
            DEPENDS ${src_file}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            VERBATIM COMMAND_EXPAND_LISTS
        )

        list(APPEND deployed_files ${output_file})
    endforeach()

    add_custom_target(DeployRealESrganModels DEPENDS ${deployed_files})
endif()

set_target_properties(DeployRealESrganModels PROPERTIES FOLDER "External")
