add_custom_command(OUTPUT PythonVenv/pyvenv.cfg
    COMMAND Setup.bat
    COMMENT "Seting up Python virtual environment..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR})
else()
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

add_custom_target(DeployPython ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}/Python/Lib
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/PythonVenv/Lib/site-packages ${output_dir}/Python/Lib/site-packages
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(DeployPython PROPERTIES FOLDER "External")
