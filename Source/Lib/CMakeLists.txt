set(camera_database_src "${PROJECT_SOURCE_DIR}/External/openMVG/openMVG/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt")
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR})
else()
    set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()
set(camera_database_dst "${output_dir}/sensor_width_camera_database.txt")
add_custom_command(OUTPUT ${camera_database_dst}
    COMMAND ${CMAKE_COMMAND} -E copy ${camera_database_src} ${camera_database_dst}
    COMMENT "Copying camera database ..."
    MAIN_DEPENDENCY ${camera_database_src}
    DEPENDS ${camera_database_src}
    VERBATIM COMMAND_EXPAND_LISTS
)

set(aihi_files
    Include/AIHoloImager/AIHoloImager.hpp
    Include/AIHoloImager/Mesh.hpp
    Include/AIHoloImager/Texture.hpp
    Source/AIHoloImager.cpp
    Source/Mesh.cpp
    Source/Texture.cpp
    Source/Gpu/GpuBuffer.cpp
    Source/Gpu/GpuBuffer.hpp
    Source/Gpu/GpuBufferHelper.hpp
    Source/Gpu/GpuCommandList.cpp
    Source/Gpu/GpuCommandList.hpp
    Source/Gpu/GpuDescriptorAllocator.cpp
    Source/Gpu/GpuDescriptorAllocator.hpp
    Source/Gpu/GpuDescriptorHeap.cpp
    Source/Gpu/GpuDescriptorHeap.hpp
    Source/Gpu/GpuMemoryAllocator.cpp
    Source/Gpu/GpuMemoryAllocator.hpp
    Source/Gpu/GpuResourceViews.cpp
    Source/Gpu/GpuResourceViews.hpp
    Source/Gpu/GpuSystem.cpp
    Source/Gpu/GpuSystem.hpp
    Source/Gpu/GpuTexture.cpp
    Source/Gpu/GpuTexture.hpp
    Source/MaskGen/MaskGenerator.cpp
    Source/MaskGen/MaskGenerator.py
    Source/MaskGen/MaskGenerator.hpp
    Source/MeshRecon/MeshReconstruction.cpp
    Source/MeshRecon/MeshReconstruction.hpp
    Source/MvRenderer/MultiViewRenderer.cpp
    Source/MvRenderer/MultiViewRenderer.hpp
    Source/MvRenderer/Shader/DownsampleCs.hlsl
    Source/MvRenderer/Shader/RenderVs.hlsl
    Source/MvRenderer/Shader/RenderPs.hlsl
    Source/SfM/StructureFromMotion.cpp
    Source/SfM/StructureFromMotion.hpp
    Source/Util/ComPtr.hpp
    Source/Util/ErrorHandling.cpp
    Source/Util/ErrorHandling.hpp
    Source/Util/Noncopyable.hpp
    Source/Util/SmartPtrHelper.hpp
    Source/Util/Uuid.hpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_files})
source_group("Source" FILES ${camera_database_src})

add_library(AIHoloImagerLib STATIC
    ${camera_database_src}
    ${aihi_files}
)

macro(AddShaderFile file_name shader_type entry_point)
    get_filename_component(file_base_name ${file_name} NAME_WE)
    set(variable_name ${file_base_name}_shader)
    set(output_name "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/CompiledShader/${file_base_name}.h")
    set(debug_option "-Zi;-Od;-Qembed_debug")
    set(release_option "-O2")

    add_custom_command(OUTPUT ${output_name}
        COMMAND dxc "$<IF:$<CONFIG:Debug>,${debug_option},${release_option}>" -T ${shader_type}_6_3 -Vn ${variable_name} -E "${entry_point}" -Fh "${output_name}" /nologo "${CMAKE_CURRENT_SOURCE_DIR}/${file_name}"
        COMMENT "Compiling ${file_name} to ${output_name}..."
        MAIN_DEPENDENCY ${file_name}
        DEPENDS ${file_name}
        VERBATIM COMMAND_EXPAND_LISTS
    )
endmacro()

AddShaderFile(Source/MvRenderer/Shader/DownsampleCs.hlsl "cs" "main")
AddShaderFile(Source/MvRenderer/Shader/RenderVs.hlsl "vs" "main")
AddShaderFile(Source/MvRenderer/Shader/RenderPs.hlsl "ps" "main")

target_include_directories(AIHoloImagerLib
    PUBLIC
        Include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/External/openMVG/openMVG/src/software/SfM/export
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)
string(REPLACE "\\" "/" AIHI_PY_STDLIB_DIR ${Python3_STDLIB})
string(REPLACE "\\" "/" AIHI_PY_RUNTIME_LIB_DIR ${Python3_RUNTIME_LIBRARY_DIRS})

target_compile_definitions(AIHoloImagerLib
    PRIVATE
        AIHI_PY_STDLIB_DIR="${AIHI_PY_STDLIB_DIR}"
        AIHI_PY_RUNTIME_LIB_DIR="${AIHI_PY_RUNTIME_LIB_DIR}"
)

target_link_libraries(AIHoloImagerLib
    PRIVATE
        assimp
        d3d12
        dxgi
        dxguid
        DirectX-Headers
        openMVG_exif
        openMVG_matching_image_collection
        openMVG_sfm
        Python3::Python
        stb
)

set(mask_generator_src "${CMAKE_CURRENT_SOURCE_DIR}/Source/MaskGen/MaskGenerator.py")
get_filename_component(mask_generator_dir ${mask_generator_src} DIRECTORY)
get_filename_component(mask_generator_file_name ${mask_generator_src} NAME)
get_filename_component(mask_generator_stem ${mask_generator_src} NAME_WE)
set(mask_generator_dst "${output_dir}/${mask_generator_stem}.pyc")
add_custom_command(OUTPUT ${mask_generator_dst}
    COMMAND ${Python3_EXECUTABLE} -m compileall -b ${mask_generator_src}
    COMMAND ${CMAKE_COMMAND} -E copy ${mask_generator_dir}/${mask_generator_stem}.pyc ${mask_generator_dst}
    COMMENT "Compiling ${mask_generator_file_name} ..."
    MAIN_DEPENDENCY ${mask_generator_src}
    DEPENDS ${mask_generator_src}
    VERBATIM COMMAND_EXPAND_LISTS
)

add_dependencies(AIHoloImagerLib DeployOpenMVS DeployPython)
