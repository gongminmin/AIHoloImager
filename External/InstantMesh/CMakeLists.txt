UpdateExternalLib("InstantMesh" "https://github.com/TencentARC/InstantMesh.git" "7fe95627cf819748f7830b2b278f302a9d798d17")

# zero123plus files need to be in source form
set(copy_files
    InstantMesh/zero123plus/model.py
    InstantMesh/zero123plus/pipeline.py
)
foreach(src_file ${copy_files})
    set(dst_file "${aihi_output_dir}/${src_file}")
    add_custom_command(OUTPUT ${dst_file}
        COMMAND ${CMAKE_COMMAND} -E copy ${src_file} ${dst_file}
        COMMENT "Copying ${src_file} ..."
        MAIN_DEPENDENCY ${src_file}
        DEPENDS ${src_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )
endforeach()

set(instant_mesh_files
    ${copy_files}
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/InstantMesh FILES ${instant_mesh_files})

add_library(InstantMesh INTERFACE
    ${instant_mesh_files}
)

set_target_properties(InstantMesh PROPERTIES FOLDER "External")
