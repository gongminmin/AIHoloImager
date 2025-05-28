# Copyright (c) 2025 Minmin Gong
#

if(is_ci_env)
    add_custom_target(DeployU2NetModels)
else()
    set(u2net_pth_file "${CMAKE_CURRENT_SOURCE_DIR}/U2NetModels/u2net.pth")
    set(u2netp_pth_file "${CMAKE_CURRENT_SOURCE_DIR}/U2NetModels/u2netp.pth")
    if((NOT EXISTS ${u2net_pth_file}) OR (NOT EXISTS ${u2netp_pth_file}))
        execute_process(
            COMMAND DownloadGd.bat
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()

    set(deployed_files )
    foreach(pth_file ${u2net_pth_file} ${u2netp_pth_file})
        get_filename_component(pth_file_name ${pth_file} NAME)
        set(output_file "${aihi_output_dir}/Models/U-2-Net/${pth_file_name}")
        add_custom_command(OUTPUT ${output_file}
            COMMAND ${CMAKE_COMMAND} -E create_hardlink ${pth_file} ${output_file}
            COMMENT "Deploying ${pth_file}..."
            DEPENDS ${pth_file}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            VERBATIM COMMAND_EXPAND_LISTS
        )

        list(APPEND deployed_files ${output_file})
    endforeach()

    add_custom_target(DeployU2NetModels DEPENDS ${deployed_files})
    add_dependencies(DeployU2NetModels DeployPython)
endif()

set_target_properties(DeployU2NetModels PROPERTIES FOLDER "External")
