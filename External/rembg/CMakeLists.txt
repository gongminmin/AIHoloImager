# Copyright (c) 2025 Minmin Gong
#

if(is_ci_env)
    add_custom_target(DeployRembgModels)
else()
    set(url "https://github.com/danielgatis/rembg/releases/download/v0.0.0/")
    set(local_dir "${CMAKE_CURRENT_SOURCE_DIR}/RembgModels/")
    DownloadFile("${url}u2net.onnx" "${local_dir}u2net.onnx")

    set(onnx_file "${local_dir}/u2net.onnx")
    set(pt_file "${local_dir}/u2net.pt")
    add_custom_command(OUTPUT ${pt_file}
        COMMAND Convert.bat
        COMMENT "Converting ${pt_file}..."
        DEPENDS ${onnx_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )

    set(pt_file "u2net.pt")
    set(output_file ${aihi_output_dir}/Models/rembg/${pt_file})
    set(pt_file RembgModels/${pt_file})
    add_custom_command(OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND} -E create_hardlink ${pt_file} ${output_file}
        COMMENT "Deploying ${pt_file}..."
        DEPENDS ${pt_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )

    add_custom_target(DeployRembgModels DEPENDS ${output_file})
    add_dependencies(DeployRembgModels DeployPython)
endif()

set_target_properties(DeployRembgModels PROPERTIES FOLDER "External")
