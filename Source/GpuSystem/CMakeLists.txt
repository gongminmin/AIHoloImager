# Copyright (c) 2025 Minmin Gong
#

if(ai_holo_imager_platform_windows AND AIHI_ENABLE_D3D12)
    set(enable_d3d12 TRUE)
endif()

set(aihi_gpu_system_files
    Include/Gpu/GpuBuffer.hpp
    Include/Gpu/GpuCommandAllocatorInfo.hpp
    Include/Gpu/GpuCommandList.hpp
    Include/Gpu/GpuConstantBuffer.hpp
    Include/Gpu/GpuFormat.hpp
    Include/Gpu/GpuMemoryAllocator.hpp
    Include/Gpu/GpuMipmapper.hpp
    Include/Gpu/GpuResource.hpp
    Include/Gpu/GpuResourceViews.hpp
    Include/Gpu/GpuSampler.hpp
    Include/Gpu/GpuShader.hpp
    Include/Gpu/GpuSystem.hpp
    Include/Gpu/GpuTexture.hpp
    Include/Gpu/GpuVertexAttrib.hpp
    Include/Gpu/InternalDefine.hpp
    Source/Internal/GpuBufferInternal.cpp
    Source/Internal/GpuBufferInternal.hpp
    Source/Internal/GpuCommandAllocatorInfoInternal.cpp
    Source/Internal/GpuCommandAllocatorInfoInternal.hpp
    Source/Internal/GpuCommandListInternal.cpp
    Source/Internal/GpuCommandListInternal.hpp
    Source/Internal/GpuResourceInternal.cpp
    Source/Internal/GpuResourceInternal.hpp
    Source/Internal/GpuResourceViewsInternal.cpp
    Source/Internal/GpuResourceViewsInternal.hpp
    Source/Internal/GpuSamplerInternal.cpp
    Source/Internal/GpuSamplerInternal.hpp
    Source/Internal/GpuShaderInternal.cpp
    Source/Internal/GpuShaderInternal.hpp
    Source/Internal/GpuSystemInternal.cpp
    Source/Internal/GpuSystemInternal.hpp
    Source/Internal/GpuSystemInternalFactory.cpp
    Source/Internal/GpuSystemInternalFactory.hpp
    Source/Internal/GpuTextureInternal.cpp
    Source/Internal/GpuTextureInternal.hpp
    Source/Internal/GpuVertexAttribInternal.cpp
    Source/Internal/GpuVertexAttribInternal.hpp
    Source/Shader/GenMipmapCs.hlsl
    Source/Shader/Platform.hlslh
    Source/GpuBuffer.cpp
    Source/GpuCommandAllocatorInfo.cpp
    Source/GpuCommandList.cpp
    Source/GpuConstantBuffer.cpp
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
    Source/InternalImp.hpp
    Source/pch.hpp
)
if(enable_d3d12)
    set(aihi_gpu_system_files ${aihi_gpu_system_files}
        Include/Gpu/D3D12/D3D12Traits.hpp
        Source/Internal/D3D12/D3D12Buffer.cpp
        Source/Internal/D3D12/D3D12Buffer.hpp
        Source/Internal/D3D12/D3D12CommandAllocatorInfo.cpp
        Source/Internal/D3D12/D3D12CommandAllocatorInfo.hpp
        Source/Internal/D3D12/D3D12CommandList.cpp
        Source/Internal/D3D12/D3D12CommandList.hpp
        Source/Internal/D3D12/D3D12Conversion.cpp
        Source/Internal/D3D12/D3D12Conversion.hpp
        Source/Internal/D3D12/D3D12DescriptorAllocator.cpp
        Source/Internal/D3D12/D3D12DescriptorAllocator.hpp
        Source/Internal/D3D12/D3D12DescriptorHeap.cpp
        Source/Internal/D3D12/D3D12DescriptorHeap.hpp
        Source/Internal/D3D12/D3D12ImpDefine.hpp
        Source/Internal/D3D12/D3D12Resource.cpp
        Source/Internal/D3D12/D3D12Resource.hpp
        Source/Internal/D3D12/D3D12ResourceViews.cpp
        Source/Internal/D3D12/D3D12ResourceViews.hpp
        Source/Internal/D3D12/D3D12Sampler.cpp
        Source/Internal/D3D12/D3D12Sampler.hpp
        Source/Internal/D3D12/D3D12Shader.cpp
        Source/Internal/D3D12/D3D12Shader.hpp
        Source/Internal/D3D12/D3D12System.cpp
        Source/Internal/D3D12/D3D12System.hpp
        Source/Internal/D3D12/D3D12Texture.cpp
        Source/Internal/D3D12/D3D12Texture.hpp
        Source/Internal/D3D12/D3D12Util.cpp
        Source/Internal/D3D12/D3D12Util.hpp
        Source/Internal/D3D12/D3D12VertexAttrib.cpp
        Source/Internal/D3D12/D3D12VertexAttrib.hpp
    )
endif()
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_gpu_system_files})

AddShaderFile(Source/Shader/GenMipmapCs.hlsl "cs" "main")

add_library(AIHoloImagerGpuSystem STATIC
    ${aihi_gpu_system_files}
)

if(enable_d3d12)
    target_compile_definitions(AIHoloImagerGpuSystem
        PUBLIC
            AIHI_ENABLE_D3D12
    )
endif()

target_include_directories(AIHoloImagerGpuSystem
    PUBLIC
        Include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

target_precompile_headers(AIHoloImagerGpuSystem
    PRIVATE
        Source/pch.hpp
)

target_link_libraries(AIHoloImagerGpuSystem
    PRIVATE
        AIHoloImagerBase
)
if(enable_d3d12)
    target_link_libraries(AIHoloImagerGpuSystem
        PUBLIC
            DirectX-Headers

        PRIVATE
            d3d12
            dxgi
            dxguid
            dxcompiler
    )
endif()
