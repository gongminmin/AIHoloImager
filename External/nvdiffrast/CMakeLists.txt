UpdateExternalLib("nvdiffrast" "https://github.com/NVlabs/nvdiffrast.git" "c5caf7bdb8a2448acc491a9faa47753972edd380")

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR})
else()
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

add_custom_target(DeployNvDiffrast ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/nvdiffrast/nvdiffrast ${output_dir}/nvdiffrast
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(DeployNvDiffrast PROPERTIES FOLDER "External")
