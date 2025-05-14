# Copyright (c) 2025 Minmin Gong
#

if(is_ci_env)
    add_custom_target(DeployU2NetModels)
else()
    set(pth_file "${CMAKE_CURRENT_SOURCE_DIR}/U2NetModels/u2net.pth")
    if(NOT EXISTS ${pth_file})
        execute_process(
            COMMAND DownloadGd.bat
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()

    set(output_file "${aihi_output_dir}/Models/U-2-Net/u2net.pth")
    add_custom_command(OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND} -E create_hardlink ${pth_file} ${output_file}
        COMMENT "Deploying ${pth_file}..."
        DEPENDS ${pth_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )

    add_custom_target(DeployU2NetModels DEPENDS ${output_file})
    add_dependencies(DeployU2NetModels DeployPython)
endif()

set_target_properties(DeployU2NetModels PROPERTIES FOLDER "External")
