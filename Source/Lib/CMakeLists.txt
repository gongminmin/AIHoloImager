# Copyright (c) 2025 Minmin Gong
#

set(camera_database_src "${PROJECT_SOURCE_DIR}/External/openMVG/openMVG/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt")
set(camera_database_dst "${aihi_output_dir}/CameraDatabase.dat")
add_custom_command(OUTPUT ${camera_database_dst}
    COMMAND ${Python3_EXECUTABLE} ConvertDatasheet.py ${camera_database_src} ${camera_database_dst}
    COMMENT "Converting camera database ..."
    MAIN_DEPENDENCY ${camera_database_src}
    DEPENDS ${camera_database_src}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Source/SfM
    VERBATIM COMMAND_EXPAND_LISTS
)

set(aihi_files
    Include/AIHoloImager/AIHoloImager.hpp
    Include/AIHoloImager/ElementFormat.hpp
    Include/AIHoloImager/Mesh.hpp
    Include/AIHoloImager/Texture.hpp
    Source/AIHoloImager.cpp
    Source/AIHoloImagerInternal.hpp
    Source/ElementFormat.cpp
    Source/Mesh.cpp
    Source/Texture.cpp
    Source/pch.hpp
    Source/Delighter/Shader/MergeMaskCs.hlsl
    Source/Delighter/Delighter.cpp
    Source/Delighter/Delighter.hpp
    Source/Delighter/Delighter.py
    Source/Delighter/ModMidas/Effnet/Conv2dLayers.py
    Source/Delighter/ModMidas/Effnet/EfficientNet.py
    Source/Delighter/ModMidas/Effnet/__init__.py
    Source/Delighter/ModMidas/Blocks.py
    Source/Delighter/ModMidas/MidasNet.py
    Source/Delighter/ModMidas/Wsl.py
    Source/Delighter/ModMidas/__init__.py
    Source/MaskGen/Shader/CalcBBoxCs.hlsl
    Source/MaskGen/Shader/DownsampleCs.hlsl
    Source/MaskGen/Shader/ErosionDilationCs.hlsl
    Source/MaskGen/Shader/GaussianBlurCs.hlsl
    Source/MaskGen/Shader/MergeMaskCs.hlsl
    Source/MaskGen/Shader/NormalizeImageCs.hlsl
    Source/MaskGen/Shader/UpsampleCs.hlsl
    Source/MaskGen/Shader/StatImageCs.hlsl
    Source/MaskGen/Shader/StatPredCs.hlsl
    Source/MaskGen/MaskGenerator.cpp
    Source/MaskGen/MaskGenerator.hpp
    Source/MaskGen/MaskGenerator.py
    Source/MaskGen/U2Net.py
    Source/MeshGen/Shader/InvisibleFacesRemover/AccumFacesCs.hlsl
    Source/MeshGen/Shader/InvisibleFacesRemover/FaceIdPs.hlsl
    Source/MeshGen/Shader/InvisibleFacesRemover/FaceIdVs.hlsl
    Source/MeshGen/Shader/InvisibleFacesRemover/FilterFacesCs.hlsl
    Source/MeshGen/Shader/InvisibleFacesRemover/MarkFacesCs.hlsl
    Source/MeshGen/Shader/MarchingCubes/CalcCubeIndicesCs.hlsl
    Source/MeshGen/Shader/MarchingCubes/GenVerticesIndicesCs.hlsl
    Source/MeshGen/Shader/MarchingCubes/MarchingCubesUtil.hlslh
    Source/MeshGen/Shader/MarchingCubes/ProcessNonEmptyCubesCs.hlsl
    Source/MeshGen/Shader/ApplyVertexColorCs.hlsl
    Source/MeshGen/Shader/Dilate3DCs.hlsl
    Source/MeshGen/Shader/DilateCs.hlsl
    Source/MeshGen/Shader/ErosionMaskCs.hlsl
    Source/MeshGen/Shader/ExtractMaskCs.hlsl
    Source/MeshGen/Shader/GatherVolumeCs.hlsl
    Source/MeshGen/Shader/MergeTextureCs.hlsl
    Source/MeshGen/Shader/ResizeCs.hlsl
    Source/MeshGen/Shader/RotatePs.hlsl
    Source/MeshGen/Shader/RotateVs.hlsl
    Source/MeshGen/Shader/ScatterIndexCs.hlsl
    Source/MeshGen/DiffOptimizer.cpp
    Source/MeshGen/DiffOptimizer.hpp
    Source/MeshGen/DiffOptimizer.py
    Source/MeshGen/InvisibleFacesRemover.cpp
    Source/MeshGen/InvisibleFacesRemover.hpp
    Source/MeshGen/MarchingCubes.cpp
    Source/MeshGen/MarchingCubes.hpp
    Source/MeshGen/MeshGenerator.cpp
    Source/MeshGen/MeshGenerator.hpp
    Source/MeshGen/MeshGenerator.py
    Source/MeshGen/Trellis/Models/StructuredLatentVae/Base.py
    Source/MeshGen/Trellis/Models/StructuredLatentVae/DecoderVolume.py
    Source/MeshGen/Trellis/Models/StructuredLatentVae/__init__.py
    Source/MeshGen/Trellis/Models/SparseStructureFlow.py
    Source/MeshGen/Trellis/Models/SparseStructureVae.py
    Source/MeshGen/Trellis/Models/StructuredLatentFlow.py
    Source/MeshGen/Trellis/Models/__init__.py
    Source/MeshGen/Trellis/Modules/Attention/FullAttn.py
    Source/MeshGen/Trellis/Modules/Attention/Modules.py
    Source/MeshGen/Trellis/Modules/Attention/__init__.py
    Source/MeshGen/Trellis/Modules/Sparse/Attention/FullAttn.py
    Source/MeshGen/Trellis/Modules/Sparse/Attention/Modules.py
    Source/MeshGen/Trellis/Modules/Sparse/Attention/WindowedAttn.py
    Source/MeshGen/Trellis/Modules/Sparse/Attention/__init__.py
    Source/MeshGen/Trellis/Modules/Sparse/Conv/ConvSpconv.py
    Source/MeshGen/Trellis/Modules/Sparse/Conv/__init__.py
    Source/MeshGen/Trellis/Modules/Sparse/Transformer/Blocks.py
    Source/MeshGen/Trellis/Modules/Sparse/Transformer/Modulated.py
    Source/MeshGen/Trellis/Modules/Sparse/Transformer/__init__.py
    Source/MeshGen/Trellis/Modules/Sparse/Basic.py
    Source/MeshGen/Trellis/Modules/Sparse/Linear.py
    Source/MeshGen/Trellis/Modules/Sparse/Nonlinearity.py
    Source/MeshGen/Trellis/Modules/Sparse/Norm.py
    Source/MeshGen/Trellis/Modules/Sparse/Spatial.py
    Source/MeshGen/Trellis/Modules/Sparse/__init__.py
    Source/MeshGen/Trellis/Modules/Transformer/Blocks.py
    Source/MeshGen/Trellis/Modules/Transformer/Modulated.py
    Source/MeshGen/Trellis/Modules/Transformer/__init__.py
    Source/MeshGen/Trellis/Modules/Norm.py
    Source/MeshGen/Trellis/Modules/Spatial.py
    Source/MeshGen/Trellis/Modules/Utils.py
    Source/MeshGen/Trellis/Pipelines/Samplers/ClassifierFreeGuidanceMixin.py
    Source/MeshGen/Trellis/Pipelines/Samplers/FlowEuler.py
    Source/MeshGen/Trellis/Pipelines/Samplers/GuidanceIntervalMixin.py
    Source/MeshGen/Trellis/Pipelines/Samplers/__init__.py
    Source/MeshGen/Trellis/Pipelines/TrellisImageTo3D.py
    Source/MeshGen/Trellis/Pipelines/__init__.py
    Source/MeshSimp/MeshSimplification.cpp
    Source/MeshSimp/MeshSimplification.hpp
    Source/Python/PythonSystem.cpp
    Source/Python/PythonSystem.hpp
    Source/Python/PythonSystem.py
    Source/SfM/MoGe/Geometry.py
    Source/SfM/MoGe/MoGeModel.py
    Source/SfM/MoGe/__init__.py
    Source/SfM/Shader/UndistortCs.hlsl
    Source/SfM/ConvertDatasheet.py
    Source/SfM/PointCloudEstimator.py
    Source/SfM/StructureFromMotion.cpp
    Source/SfM/StructureFromMotion.hpp
    Source/TextureRecon/Shader/FlattenVs.hlsl
    Source/TextureRecon/Shader/FlattenPs.hlsl
    Source/TextureRecon/Shader/GenShadowMapVs.hlsl
    Source/TextureRecon/Shader/ProjectTextureCs.hlsl
    Source/TextureRecon/Shader/ResolveTextureCs.hlsl
    Source/TextureRecon/TextureReconstruction.cpp
    Source/TextureRecon/TextureReconstruction.hpp
    Source/Util/Shader/Lanczos.hlslh
    Source/Util/Shader/Nn.hlslh
    Source/Util/Shader/Utils.hlslh
    Source/Util/BoundingBox.cpp
    Source/Util/BoundingBox.hpp
    Source/Util/FormatConversion.cpp
    Source/Util/FormatConversion.hpp
    Source/Util/PerfProfiler.cpp
    Source/Util/PerfProfiler.hpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_files})
source_group("Source" FILES ${camera_database_src})

add_library(AIHoloImagerLib STATIC
    ${camera_database_src}
    ${aihi_files}
)

AddShaderFile(Source/Delighter/Shader/MergeMaskCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/CalcBBoxCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/DownsampleCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/ErosionDilationCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/GaussianBlurCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/MergeMaskCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/NormalizeImageCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/UpsampleCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/StatImageCs.hlsl "cs" "main")
AddShaderFile(Source/MaskGen/Shader/StatPredCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/InvisibleFacesRemover/AccumFacesCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/InvisibleFacesRemover/FaceIdVs.hlsl "vs" "main")
AddShaderFile(Source/MeshGen/Shader/InvisibleFacesRemover/FaceIdPs.hlsl "ps" "main")
AddShaderFile(Source/MeshGen/Shader/InvisibleFacesRemover/FilterFacesCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/InvisibleFacesRemover/MarkFacesCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/MarchingCubes/CalcCubeIndicesCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/MarchingCubes/GenVerticesIndicesCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/MarchingCubes/ProcessNonEmptyCubesCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/ApplyVertexColorCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/Dilate3DCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/DilateCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/ErosionMaskCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/GatherVolumeCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/MergeTextureCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/ResizeCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/RotateVs.hlsl "vs" "main")
AddShaderFile(Source/MeshGen/Shader/RotatePs.hlsl "ps" "main")
AddShaderFile(Source/MeshGen/Shader/ScatterIndexCs.hlsl "cs" "main")
AddShaderFile(Source/SfM/Shader/UndistortCs.hlsl "cs" "main")
AddShaderFile(Source/TextureRecon/Shader/FlattenVs.hlsl "vs" "main")
AddShaderFile(Source/TextureRecon/Shader/FlattenPs.hlsl "ps" "main")
AddShaderFile(Source/TextureRecon/Shader/GenShadowMapVs.hlsl "vs" "main")
AddShaderFile(Source/TextureRecon/Shader/ProjectTextureCs.hlsl "cs" "main")
AddShaderFile(Source/TextureRecon/Shader/ResolveTextureCs.hlsl "cs" "main")
AddShaderFile(Source/MeshGen/Shader/ExtractMaskCs.hlsl "cs" "main")

target_include_directories(AIHoloImagerLib
    PUBLIC
        Include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/External/openMVG/openMVG/src/software/SfM/export
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
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
        glm
        openMVG_exif
        openMVG_matching_image_collection
        openMVG_sfm
        Python3::Python
        stb
        xatlas
        AIHoloImagerBase
        AIHoloImagerGpuSystem
        AIHoloImagerTensorConverter
)

set(py_files
    Source/Delighter/Delighter.py
    Source/Delighter/ModMidas/Effnet/Conv2dLayers.py
    Source/Delighter/ModMidas/Effnet/EfficientNet.py
    Source/Delighter/ModMidas/Effnet/__init__.py
    Source/Delighter/ModMidas/Blocks.py
    Source/Delighter/ModMidas/MidasNet.py
    Source/Delighter/ModMidas/Wsl.py
    Source/Delighter/ModMidas/__init__.py
    Source/MaskGen/MaskGenerator.py
    Source/MaskGen/U2Net.py
    Source/MeshGen/MeshGenerator.py
    Source/MeshGen/Trellis/Models/StructuredLatentVae/Base.py
    Source/MeshGen/Trellis/Models/StructuredLatentVae/DecoderVolume.py
    Source/MeshGen/Trellis/Models/StructuredLatentVae/Encoder.py
    Source/MeshGen/Trellis/Models/StructuredLatentVae/__init__.py
    Source/MeshGen/Trellis/Models/SparseStructureFlow.py
    Source/MeshGen/Trellis/Models/SparseStructureVae.py
    Source/MeshGen/Trellis/Models/StructuredLatentFlow.py
    Source/MeshGen/Trellis/Models/__init__.py
    Source/MeshGen/Trellis/Modules/Attention/FullAttn.py
    Source/MeshGen/Trellis/Modules/Attention/Modules.py
    Source/MeshGen/Trellis/Modules/Attention/__init__.py
    Source/MeshGen/Trellis/Modules/Sparse/Attention/FullAttn.py
    Source/MeshGen/Trellis/Modules/Sparse/Attention/Modules.py
    Source/MeshGen/Trellis/Modules/Sparse/Attention/WindowedAttn.py
    Source/MeshGen/Trellis/Modules/Sparse/Attention/__init__.py
    Source/MeshGen/Trellis/Modules/Sparse/Conv/ConvSpconv.py
    Source/MeshGen/Trellis/Modules/Sparse/Conv/__init__.py
    Source/MeshGen/Trellis/Modules/Sparse/Transformer/Blocks.py
    Source/MeshGen/Trellis/Modules/Sparse/Transformer/Modulated.py
    Source/MeshGen/Trellis/Modules/Sparse/Transformer/__init__.py
    Source/MeshGen/Trellis/Modules/Sparse/Basic.py
    Source/MeshGen/Trellis/Modules/Sparse/Linear.py
    Source/MeshGen/Trellis/Modules/Sparse/Nonlinearity.py
    Source/MeshGen/Trellis/Modules/Sparse/Norm.py
    Source/MeshGen/Trellis/Modules/Sparse/Spatial.py
    Source/MeshGen/Trellis/Modules/Sparse/__init__.py
    Source/MeshGen/Trellis/Modules/Transformer/Blocks.py
    Source/MeshGen/Trellis/Modules/Transformer/Modulated.py
    Source/MeshGen/Trellis/Modules/Transformer/__init__.py
    Source/MeshGen/Trellis/Modules/Norm.py
    Source/MeshGen/Trellis/Modules/Spatial.py
    Source/MeshGen/Trellis/Modules/Utils.py
    Source/MeshGen/Trellis/Pipelines/Samplers/ClassifierFreeGuidanceMixin.py
    Source/MeshGen/Trellis/Pipelines/Samplers/FlowEuler.py
    Source/MeshGen/Trellis/Pipelines/Samplers/GuidanceIntervalMixin.py
    Source/MeshGen/Trellis/Pipelines/Samplers/__init__.py
    Source/MeshGen/Trellis/Pipelines/TrellisImageTo3D.py
    Source/MeshGen/Trellis/Pipelines/__init__.py
    Source/MeshGen/DiffOptimizer.py
    Source/Python/PythonSystem.py
    Source/SfM/MoGe/Geometry.py
    Source/SfM/MoGe/MoGeModel.py
    Source/SfM/MoGe/__init__.py
    Source/SfM/PointCloudEstimator.py
)
foreach(py_file ${py_files})
    string(REPLACE "/" ";" path_list ${py_file})
    list(REMOVE_AT path_list 0 1)
    list(REMOVE_AT path_list -1)
    string(REPLACE ";" "/" rel_dir "${path_list}")
    get_filename_component(py_file_dir ${py_file} DIRECTORY)
    get_filename_component(py_file_name ${py_file} NAME)
    get_filename_component(py_file_stem ${py_file} NAME_WE)
    set(py_file_dst "${aihi_output_dir}/${rel_dir}/${py_file_stem}.pyc")
    add_custom_command(OUTPUT ${py_file_dst}
        COMMAND ${Python3_EXECUTABLE} -m compileall -b -o $<IF:$<CONFIG:Debug>,0,1> ${py_file}
        COMMAND ${CMAKE_COMMAND} -E copy ${py_file_dir}/${py_file_stem}.pyc ${py_file_dst}
        COMMENT "Compiling ${rel_dir}/${py_file_name} ..."
        MAIN_DEPENDENCY ${py_file}
        DEPENDS ${py_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM COMMAND_EXPAND_LISTS
    )
endforeach()

add_dependencies(AIHoloImagerLib DeployPython DeployU2NetModels DeployIntrinsicModels DeployTrellisModels DeployMoGeModels AIHoloImagerGpuDiffRender AIHoloImagerSubMConv)

if(MSVC)
    target_link_libraries(AIHoloImagerLib
        PUBLIC
            delayimp.lib
    )
    target_link_options(AIHoloImagerLib
        PUBLIC
            "/DELAYLOAD:$<TARGET_FILE_NAME:AIHoloImagerTensorConverter>"
    )
endif()
