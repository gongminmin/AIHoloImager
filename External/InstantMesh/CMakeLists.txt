UpdateExternalLib("InstantMesh" "https://github.com/TencentARC/InstantMesh.git" "7fe95627cf819748f7830b2b278f302a9d798d17")

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR})
else()
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

add_custom_target(DeployInstantMesh ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/InstantMesh ${output_dir}/InstantMesh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(DeployInstantMesh PROPERTIES FOLDER "External")
