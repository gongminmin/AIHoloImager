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

set(shader_targets )
set(shader_compile_options )
if(ai_holo_imager_platform_windows AND AIHI_ENABLE_D3D12)
    list(APPEND shader_targets "dxil")
    list(APPEND shader_compile_options "-DD3D12_SHADER")
endif()
if(AIHI_ENABLE_VULKAN)
    list(APPEND shader_targets "spv")
    list(APPEND shader_compile_options "-spirv -fspv-target-env=vulkan1.3 -DVULKAN_SHADER")
endif()
list(LENGTH shader_targets num_shader_targets)
if(num_shader_targets EQUAL 0)
    message(FATAL_ERROR "At least one GPU API must be enabled.")
endif()

macro(AddShaderFile file_name shader_type entry_point)
    get_filename_component(file_base_name ${file_name} NAME_WE)
    string(REPLACE "/" ";" path_list ${file_name})
    list(REMOVE_AT path_list 0 2)
    list(REMOVE_AT path_list -1)
    string(REPLACE ";" "/" rel_dir "${path_list}")
    set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/CompiledShader/${rel_dir}")
    set(common_header "${CMAKE_CURRENT_BINARY_DIR}/CompiledShader/${rel_dir}/${file_base_name}.h")

    set(debug_option "-Zi;-Od;-Qembed_debug")
    set(release_option "-O2")

    math(EXPR max_range "${num_shader_targets}-1")

    set(shader_output_names )
    foreach(i RANGE ${max_range})
        list(GET shader_targets ${i} curr_target)

        set(output_name "${output_dir}/${curr_target}/${file_base_name}.h")
        if(i EQUAL 0)
            file(WRITE ${common_header} "#pragma once\n")
        endif()
        file(APPEND ${common_header} "#include \"CompiledShader/${rel_dir}/${curr_target}/${file_base_name}.h\"\n")
        list(APPEND shader_output_names ${output_name})
    endforeach()

    foreach(i RANGE ${max_range})
        list(GET shader_targets ${i} curr_target)
        list(GET shader_compile_options ${i} curr_compile_option)
        separate_arguments(curr_compile_option WINDOWS_COMMAND "${curr_compile_option}")

        set(output_name "${output_dir}/${curr_target}/${file_base_name}.h")
        set(variable_name ${file_base_name}_${curr_target})
        set(extra_opts )
        if(i EQUAL 0)
            set(extra_opts VERBATIM COMMAND_EXPAND_LISTS)
        else()
            set(extra_opts APPEND)
        endif()
        add_custom_command(OUTPUT ${shader_output_names}
            COMMAND "${AIHI_DXC_CMD}"
                "$<IF:$<CONFIG:Debug>,${debug_option},${release_option}>"
                -HV 2021 -enable-16bit-types
                -T ${shader_type}_6_3 -Vn ${variable_name} -E "${entry_point}" -Fh "${output_name}" /nologo
                "${CMAKE_CURRENT_SOURCE_DIR}/${file_name}"
                -I "${PROJECT_SOURCE_DIR}/Source/Lib/Source/Util/Shader/"
                -I "${PROJECT_SOURCE_DIR}/Source/GpuSystem/Source/Shader"
                ${curr_compile_option}
            COMMENT "Compiling ${file_name}..."
            MAIN_DEPENDENCY ${file_name}
            DEPENDS ${file_name} ${common_header}
            ${extra_opts}
        )
    endforeach()
endmacro()

add_subdirectory(Base)
add_subdirectory(GpuSystem)
add_subdirectory(GpuDiffRender)
add_subdirectory(TensorConverter)
add_subdirectory(SubMConv)
add_subdirectory(Lib)
add_subdirectory(App)
