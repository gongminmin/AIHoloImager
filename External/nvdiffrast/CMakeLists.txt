# Copyright (c) 2025 Minmin Gong
#

UpdateExternalLib("nvdiffrast" "https://github.com/NVlabs/nvdiffrast.git" "c5caf7bdb8a2448acc491a9faa47753972edd380")

set(CMAKE_CUDA_ARCHITECTURES "86;89")
enable_language(CUDA)

find_package(CUDAToolkit)

set(cuda_source_files
    nvdiffrast/nvdiffrast/common/cudaraster/impl/BinRaster.inl
    nvdiffrast/nvdiffrast/common/cudaraster/impl/Buffer.cpp
    nvdiffrast/nvdiffrast/common/cudaraster/impl/Buffer.hpp
    nvdiffrast/nvdiffrast/common/cudaraster/impl/CoarseRaster.inl
    nvdiffrast/nvdiffrast/common/cudaraster/impl/Constants.hpp
    nvdiffrast/nvdiffrast/common/cudaraster/impl/CudaRaster.cpp
    nvdiffrast/nvdiffrast/common/cudaraster/impl/Defs.hpp
    nvdiffrast/nvdiffrast/common/cudaraster/impl/FineRaster.inl
    nvdiffrast/nvdiffrast/common/cudaraster/impl/PrivateDefs.hpp
    nvdiffrast/nvdiffrast/common/cudaraster/impl/RasterImpl.cpp
    nvdiffrast/nvdiffrast/common/cudaraster/impl/RasterImpl.cu
    nvdiffrast/nvdiffrast/common/cudaraster/impl/RasterImpl.hpp
    nvdiffrast/nvdiffrast/common/cudaraster/impl/TriangleSetup.inl
    nvdiffrast/nvdiffrast/common/cudaraster/impl/Util.inl
    nvdiffrast/nvdiffrast/common/cudaraster/CudaRaster.hpp
    nvdiffrast/nvdiffrast/common/antialias.cu
    nvdiffrast/nvdiffrast/common/antialias.h
    nvdiffrast/nvdiffrast/common/common.cpp
    nvdiffrast/nvdiffrast/common/common.h
    nvdiffrast/nvdiffrast/common/framework.h
    nvdiffrast/nvdiffrast/common/interpolate.cu
    nvdiffrast/nvdiffrast/common/interpolate.h
    nvdiffrast/nvdiffrast/common/rasterize.cu
    nvdiffrast/nvdiffrast/common/rasterize.h
    nvdiffrast/nvdiffrast/common/texture.cpp
    nvdiffrast/nvdiffrast/common/texture.cu
    nvdiffrast/nvdiffrast/common/texture.h
    nvdiffrast/nvdiffrast/torch/torch_antialias.cpp
    nvdiffrast/nvdiffrast/torch/torch_bindings.cpp
    nvdiffrast/nvdiffrast/torch/torch_common.inl
    nvdiffrast/nvdiffrast/torch/torch_interpolate.cpp
    nvdiffrast/nvdiffrast/torch/torch_rasterize.cpp
    nvdiffrast/nvdiffrast/torch/torch_texture.cpp
    nvdiffrast/nvdiffrast/torch/torch_types.h
)

add_library(nvdiffrast_plugin SHARED
    ${cuda_source_files}
)

set(gl_source_files
    nvdiffrast/nvdiffrast/common/common.cpp
    nvdiffrast/nvdiffrast/common/common.h
    nvdiffrast/nvdiffrast/common/glutil.cpp
    nvdiffrast/nvdiffrast/common/glutil.h
    nvdiffrast/nvdiffrast/common/glutil_extlist.h
    nvdiffrast/nvdiffrast/common/rasterize_gl.cpp
    nvdiffrast/nvdiffrast/common/rasterize_gl.h
    nvdiffrast/nvdiffrast/torch/torch_bindings_gl.cpp
    nvdiffrast/nvdiffrast/torch/torch_common.inl
    nvdiffrast/nvdiffrast/torch/torch_rasterize_gl.cpp
    nvdiffrast/nvdiffrast/torch/torch_types.h
)

add_library(nvdiffrast_plugin_gl SHARED
    ${gl_source_files}
)

set(plugins
    nvdiffrast_plugin
    nvdiffrast_plugin_gl
)

set(torch_dir ${CMAKE_CURRENT_SOURCE_DIR}/../PythonVenv/PythonVenv/Lib/site-packages/torch)

foreach(plugin ${plugins})
    target_include_directories(${plugin}
        PRIVATE
            ${torch_dir}/include
            ${torch_dir}/include/torch/csrc/api/include
    )

    if(MSVC)
        target_compile_options(${plugin}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/wd4244> # Ignore type conversion from `int` to `float`
                $<$<COMPILE_LANGUAGE:CXX>:/wd4251> # Ignore non dll-interface as member
                $<$<COMPILE_LANGUAGE:CXX>:/wd4267> # Ignore type conversion from `size_t` to something else
                $<$<COMPILE_LANGUAGE:CXX>:/wd4275> # Ignore non dll-interface base class
                $<$<COMPILE_LANGUAGE:CXX>:/wd4819> # Ignore character code page
        )
    endif()

    target_compile_definitions(${plugin}
        PRIVATE
            NVDR_TORCH
            TORCH_API_INCLUDE_EXTENSION_H
            TORCH_EXTENSION_NAME=${plugin}
    )

    target_link_directories(${plugin}
        PRIVATE
            ${torch_dir}/lib
    )
    target_link_libraries(${plugin}
        PRIVATE
            c10.lib
            c10_cuda.lib
            torch_cpu.lib
            torch_cuda.lib
            torch.lib
            torch_python.lib
            Python3::Python
            CUDA::cudart
    )

    set_target_properties(${plugin}
        PROPERTIES
            SUFFIX ".pyd"
    )

    add_dependencies(${plugin} DeployPython)
endforeach()

target_compile_definitions(nvdiffrast_plugin
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:_GLIBCXX_USE_CXX11_ABI=0>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_OPERATORS__>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_CONVERSIONS__>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_BFLOAT16_CONVERSIONS__>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF2_OPERATORS__>
)

target_compile_options(nvdiffrast_plugin
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=3012> # Ignore deprecations
)

if(MSVC)
    target_compile_definitions(nvdiffrast_plugin_gl
        PRIVATE
            _CRT_SECURE_NO_WARNINGS
    )

    target_compile_options(nvdiffrast_plugin
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/wd4700> # Ignore uninitialized local variable
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819> # Ignore character code page
    )
    target_compile_options(nvdiffrast_plugin_gl
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/wd4099> # Ignore RasterizeGLState first `class` then `struct`
    )
endif()

if(WIN32)
    target_link_libraries(nvdiffrast_plugin_gl
        PRIVATE
            gdi32
            opengl32
            user32
            ${CMAKE_CURRENT_SOURCE_DIR}/nvdiffrast/nvdiffrast/lib/setgpu.lib
    )
else()
    target_link_libraries(nvdiffrast_plugin_gl
        PRIVATE
            GL EGL
    )
endif()

set(py_files
    nvdiffrast/nvdiffrast/torch/ops.py
    nvdiffrast/nvdiffrast/torch/__init__.py
    nvdiffrast/nvdiffrast/__init__.py
)
foreach(src_file ${py_files})
    get_filename_component(src_file_dir ${src_file} DIRECTORY)
    get_filename_component(src_file_stem ${src_file} NAME_WE)
    set(pyc_file "${src_file_dir}/${src_file_stem}.pyc")
    string(REPLACE "/" ";" path_list ${src_file})
    list(REMOVE_AT path_list 0)
    list(REMOVE_AT path_list -1)
    string(REPLACE ";" "/" rel_dir "${path_list}")
    set(dst_file "${aihi_output_dir}/${rel_dir}/${src_file_stem}.pyc")
    add_custom_command(OUTPUT ${dst_file}
        COMMAND ${Python3_EXECUTABLE} -m compileall -b -o $<IF:$<CONFIG:Debug>,0,1> ${src_file}
        COMMAND ${CMAKE_COMMAND} -E copy ${pyc_file} ${dst_file}
        COMMENT "Compiling ${src_file} ..."
        MAIN_DEPENDENCY ${src_file}
        DEPENDS ${src_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )
endforeach()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/nvdiffrast FILES ${py_files})

add_library(nvdiffrast INTERFACE
    ${py_files}
)
add_dependencies(nvdiffrast nvdiffrast_plugin nvdiffrast_plugin_gl)

set_target_properties(nvdiffrast nvdiffrast_plugin nvdiffrast_plugin_gl PROPERTIES FOLDER "External/nvdiffrast")
