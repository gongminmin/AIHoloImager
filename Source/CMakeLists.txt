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

add_subdirectory(Lib)
add_subdirectory(App)
