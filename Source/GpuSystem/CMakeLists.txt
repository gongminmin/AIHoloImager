# Copyright (c) 2025 Minmin Gong
#

set(aihi_gpu_system_files
    Include/Gpu/D3D12/D3D12Traits.hpp
    Include/Gpu/GpuBuffer.hpp
    Include/Gpu/GpuCommandList.hpp
    Include/Gpu/GpuConstantBuffer.hpp
    Include/Gpu/GpuDescriptorAllocator.hpp
    Include/Gpu/GpuDescriptorHeap.hpp
    Include/Gpu/GpuFormat.hpp
    Include/Gpu/GpuMemoryAllocator.hpp
    Include/Gpu/GpuMipmapper.hpp
    Include/Gpu/GpuResource.hpp
    Include/Gpu/GpuResourceViews.hpp
    Include/Gpu/GpuSampler.hpp
    Include/Gpu/GpuShader.hpp
    Include/Gpu/GpuSystem.hpp
    Include/Gpu/GpuTexture.hpp
    Include/Gpu/GpuUtil.hpp
    Include/Gpu/GpuVertexAttrib.hpp
    Source/D3D12/D3D12Conversion.cpp
    Source/D3D12/D3D12Conversion.hpp
    Source/Internal/D3D12/D3D12SystemFactory.cpp
    Source/Internal/D3D12/D3D12SystemFactory.hpp
    Source/Internal/D3D12/D3D12VertexAttrib.cpp
    Source/Internal/D3D12/D3D12VertexAttrib.hpp
    Source/Internal/GpuSystemInternalFactory.cpp
    Source/Internal/GpuSystemInternalFactory.hpp
    Source/Internal/GpuVertexAttribInternal.cpp
    Source/Internal/GpuVertexAttribInternal.hpp
    Source/Shader/GenMipmapCs.hlsl
    Source/GpuBuffer.cpp
    Source/GpuCommandList.cpp
    Source/GpuConstantBuffer.cpp
    Source/GpuDescriptorAllocator.cpp
    Source/GpuDescriptorHeap.cpp
    Source/GpuFormat.cpp
    Source/GpuMemoryAllocator.cpp
    Source/GpuMipmapper.cpp
    Source/GpuResource.cpp
    Source/GpuResourceViews.cpp
    Source/GpuSampler.cpp
    Source/GpuShader.cpp
    Source/GpuSystem.cpp
    Source/GpuTexture.cpp
    Source/GpuVertexAttrib.cpp
    Source/pch.hpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_gpu_system_files})

AddShaderFile(Source/Shader/GenMipmapCs.hlsl "cs" "main")

add_library(AIHoloImagerGpuSystem STATIC
    ${aihi_gpu_system_files}
)

target_include_directories(AIHoloImagerGpuSystem
    PUBLIC
        Include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

target_precompile_headers(AIHoloImagerGpuSystem
    PRIVATE
        Source/pch.hpp
)

target_link_libraries(AIHoloImagerGpuSystem
    PUBLIC
        d3d12
        dxgi
        dxguid
        DirectX-Headers
        AIHoloImagerBase
)
