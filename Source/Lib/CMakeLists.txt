set(camera_database_src "${PROJECT_SOURCE_DIR}/External/openMVG/openMVG/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt")
set(camera_database_dst "${aihi_output_dir}/sensor_width_camera_database.txt")
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
    Source/pch.hpp
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
    Source/Gpu/GpuShader.cpp
    Source/Gpu/GpuShader.hpp
    Source/Gpu/GpuSystem.cpp
    Source/Gpu/GpuSystem.hpp
    Source/Gpu/GpuTexture.cpp
    Source/Gpu/GpuTexture.hpp
    Source/Gpu/GpuUtil.hpp
    Source/MaskGen/Shader/CalcBBoxCs.hlsl
    Source/MaskGen/Shader/CalcRoiCs.hlsl
    Source/MaskGen/Shader/DownsampleCs.hlsl
    Source/MaskGen/Shader/ErosionDilationCs.hlsl
    Source/MaskGen/Shader/GaussianBlurCs.hlsl
    Source/MaskGen/Shader/Lanczos.hlslh
    Source/MaskGen/Shader/MergeMaskCs.hlsl
    Source/MaskGen/Shader/NormalizeImageCs.hlsl
    Source/MaskGen/Shader/UpsampleCs.hlsl
    Source/MaskGen/Shader/StatImageCs.hlsl
    Source/MaskGen/Shader/StatPredCs.hlsl
    Source/MaskGen/MaskGenerator.cpp
    Source/MaskGen/MaskGenerator.hpp
    Source/MaskGen/MaskGenerator.py
    Source/MeshGen/Shader/CalcCubeIndicesCs.hlsl
    Source/MeshGen/Shader/DeformationNnCs.hlsl
    Source/MeshGen/Shader/DensityNnCs.hlsl
    Source/MeshGen/Shader/DilateCs.hlsl
    Source/MeshGen/Shader/GenVerticesIndicesCs.hlsl
    Source/MeshGen/Shader/MarchingCubesUtil.hlslh
    Source/MeshGen/Shader/MergeTextureCs.hlsl
    Source/MeshGen/Shader/ProcessNonEmptyCubesCs.hlsl
    Source/MeshGen/Shader/TriplaneNeRF.hlslh
    Source/MeshGen/MarchingCubes.cpp
    Source/MeshGen/MarchingCubes.hpp
    Source/MeshGen/MeshGenerator.cpp
    Source/MeshGen/MeshGenerator.hpp
    Source/MeshGen/MeshGenerator.py
    Source/MeshGen/Lrm.py
    Source/MeshRecon/MeshReconstruction.cpp
    Source/MeshRecon/MeshReconstruction.hpp
    Source/MeshSimp/MeshSimplification.cpp
    Source/MeshSimp/MeshSimplification.hpp
    Source/MvDiffusion/MultiViewDiffusion.cpp
    Source/MvDiffusion/MultiViewDiffusion.hpp
    Source/MvDiffusion/MultiViewDiffusion.py
    Source/MvRenderer/MultiViewRenderer.cpp
    Source/MvRenderer/MultiViewRenderer.hpp
    Source/MvRenderer/Shader/BlendCs.hlsl
    Source/MvRenderer/Shader/CalcDiffusionBoxCs.hlsl
    Source/MvRenderer/Shader/CalcRenderedBoxCs.hlsl
    Source/MvRenderer/Shader/DownsampleCs.hlsl
    Source/MvRenderer/Shader/RenderVs.hlsl
    Source/MvRenderer/Shader/RenderPs.hlsl
    Source/Python/PythonSystem.cpp
    Source/Python/PythonSystem.hpp
    Source/SfM/Shader/UndistortCs.hlsl
    Source/SfM/StructureFromMotion.cpp
    Source/SfM/StructureFromMotion.hpp
    Source/TextureRecon/Shader/FlattenVs.hlsl
    Source/TextureRecon/Shader/FlattenPs.hlsl
    Source/TextureRecon/Shader/GenShadowMapVs.hlsl
    Source/TextureRecon/Shader/ProjectTextureCs.hlsl
    Source/TextureRecon/Shader/ResolveTextureCs.hlsl
    Source/TextureRecon/TextureReconstruction.cpp
    Source/TextureRecon/TextureReconstruction.hpp
    Source/Util/Shader/Nn.hlslh
    Source/Util/ComPtr.hpp
    Source/Util/ErrorHandling.cpp
    Source/Util/ErrorHandling.hpp
    Source/Util/Noncopyable.hpp
    Source/Util/BoundingBox.cpp
    Source/Util/BoundingBox.hpp
    Source/Util/SmartPtrHelper.hpp
    Source/Util/Timer.cpp
    Source/Util/Timer.hpp
    Source/Util/Util.py
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
    get_filename_component(file_dir ${file_name} DIRECTORY)
    string(REPLACE "/" ";" file_dir_list ${file_dir})
    list(GET file_dir_list 1 file_dir)
    set(variable_name ${file_base_name}_shader)
    set(output_name "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/CompiledShader/${file_dir}/${file_base_name}.h")
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

AddShaderFile(Source/MaskGen/Shader/CalcBBoxCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/CalcRoiCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/DownsampleCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/ErosionDilationCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/GaussianBlurCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/MergeMaskCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/NormalizeImageCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/UpsampleCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/StatImageCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/StatPredCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/CalcCubeIndicesCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/DeformationNnCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/DensityNnCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/DilateCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/GenVerticesIndicesCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/MergeTextureCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/ProcessNonEmptyCubesCs.hlsl "cs" "main")
AddShaderFile(Source/MvRenderer/Shader/BlendCs.hlsl "cs" "main")
AddShaderFile(Source/MvRenderer/Shader/CalcDiffusionBoxCs.hlsl "cs" "main")
AddShaderFile(Source/MvRenderer/Shader/CalcRenderedBoxCs.hlsl "cs" "main")
AddShaderFile(Source/MvRenderer/Shader/DownsampleCs.hlsl "cs" "main")
AddShaderFile(Source/MvRenderer/Shader/RenderVs.hlsl "vs" "main")
AddShaderFile(Source/MvRenderer/Shader/RenderPs.hlsl "ps" "main")
AddShaderFile(Source/SfM/Shader/UndistortCs.hlsl "cs" "main")
AddShaderFile(Source/TextureRecon/Shader/FlattenVs.hlsl "vs" "main")
AddShaderFile(Source/TextureRecon/Shader/FlattenPs.hlsl "ps" "main")
AddShaderFile(Source/TextureRecon/Shader/GenShadowMapVs.hlsl "vs" "main")
AddShaderFile(Source/TextureRecon/Shader/ProjectTextureCs.hlsl "cs" "main")
AddShaderFile(Source/TextureRecon/Shader/ResolveTextureCs.hlsl "cs" "main")

target_include_directories(AIHoloImagerLib
    PUBLIC
        Include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/External/openMVG/openMVG/src/software/SfM/export
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

string(REPLACE "\\" "/" AIHI_PY_STDLIB_DIR ${Python3_STDLIB})
string(REPLACE "\\" "/" AIHI_PY_RUNTIME_LIB_DIR ${Python3_RUNTIME_LIBRARY_DIRS})

target_compile_definitions(AIHoloImagerLib
    PRIVATE
        AIHI_PY_STDLIB_DIR="${AIHI_PY_STDLIB_DIR}"
        AIHI_PY_RUNTIME_LIB_DIR="${AIHI_PY_RUNTIME_LIB_DIR}"
)
if(AIHI_KEEP_INTERMEDIATES)
    target_compile_definitions(AIHoloImagerLib
        PUBLIC
            AIHI_KEEP_INTERMEDIATES
    )
endif()
if(ai_holo_imager_compiler_msvc)
    target_compile_definitions(AIHoloImagerLib
        PRIVATE
            _CRT_SECURE_NO_WARNINGS
    )
endif()

target_precompile_headers(AIHoloImagerLib
    PRIVATE
        Source/pch.hpp
)

target_link_libraries(AIHoloImagerLib
    PRIVATE
        assimp
        d3d12
        dxgi
        dxguid
        DirectX-Headers
        glm
        openMVG_exif
        openMVG_matching_image_collection
        openMVG_sfm
        Python3::Python
        stb
        xatlas
)

set(py_files
    Source/MaskGen/MaskGenerator.py
    Source/MeshGen/MeshGenerator.py
    Source/MeshGen/Lrm.py
    Source/MvDiffusion/MultiViewDiffusion.py
    Source/Util/Util.py
)
foreach(py_file ${py_files})
    get_filename_component(py_file_dir ${py_file} DIRECTORY)
    get_filename_component(py_file_name ${py_file} NAME)
    get_filename_component(py_file_stem ${py_file} NAME_WE)
    set(py_file_dst "${aihi_output_dir}/${py_file_stem}.pyc")
    add_custom_command(OUTPUT ${py_file_dst}
        COMMAND ${Python3_EXECUTABLE} -m compileall -b -o $<IF:$<CONFIG:Debug>,0,1> ${py_file}
        COMMAND ${CMAKE_COMMAND} -E copy ${py_file_dir}/${py_file_stem}.pyc ${py_file_dst}
        COMMENT "Compiling ${py_file_name} ..."
        MAIN_DEPENDENCY ${py_file}
        DEPENDS ${py_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )
endforeach()

add_dependencies(AIHoloImagerLib DeployOpenMVS DeployPython InstantMesh)
