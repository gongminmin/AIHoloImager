if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX /Zc:strictStrings /Zc:rvalueCast /openmp")

    if(ai_holo_imager_compiler_clangcl)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /WX")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:throwingNew /permissive- /Zc:externConstexpr")
        set(CMAKE_STATIC_LINKER_FLAGS "/WX")
        foreach(flag_var
            CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS)
            set(${flag_var} "/WX /pdbcompress")
        endforeach()
    endif()
endif()

macro(AddShaderFile file_name shader_type entry_point)
    get_filename_component(file_base_name ${file_name} NAME_WE)
    string(REPLACE "/" ";" path_list ${file_name})
    list(REMOVE_AT path_list 0 2)
    list(REMOVE_AT path_list -1)
    string(REPLACE ";" "/" rel_dir "${path_list}")
    set(variable_name ${file_base_name}_shader)
    set(output_name "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/CompiledShader/${rel_dir}/${file_base_name}.h")
    set(debug_option "-Zi;-Od;-Qembed_debug")
    set(release_option "-O2")

    add_custom_command(OUTPUT ${output_name}
        COMMAND dxc "$<IF:$<CONFIG:Debug>,${debug_option},${release_option}>" -HV 2021 -enable-16bit-types -T ${shader_type}_6_3 -Vn ${variable_name} -E "${entry_point}" -Fh "${output_name}" /nologo "${CMAKE_CURRENT_SOURCE_DIR}/${file_name}" -I "${CMAKE_CURRENT_SOURCE_DIR}/Source/Util/Shader/"
        COMMENT "Compiling ${file_name} to ${output_name}..."
        MAIN_DEPENDENCY ${file_name}
        DEPENDS ${file_name}
        VERBATIM COMMAND_EXPAND_LISTS
    )
endmacro()

add_subdirectory(Base)
add_subdirectory(GpuSystem)
add_subdirectory(GpuDiffRast)
add_subdirectory(Lib)
add_subdirectory(App)
