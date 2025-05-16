# Copyright (c) 2025 Minmin Gong
#

set(aihi_base_files
    Include/Base/ComPtr.hpp
    Include/Base/Dll.hpp
    Include/Base/Enum.hpp
    Include/Base/Errorhandling.hpp
    Include/Base/Noncopyable.hpp
    Include/Base/SmartPtrHelper.hpp
    Include/Base/Timer.hpp
    Include/Base/Uuid.hpp
    Source/Dll.cpp
    Source/Errorhandling.cpp
    Source/pch.hpp
    Source/Timer.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${aihi_base_files})

add_library(AIHoloImagerBase STATIC
    ${aihi_base_files}
)

target_include_directories(AIHoloImagerBase
    PUBLIC
        Include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

target_precompile_headers(AIHoloImagerBase
    PRIVATE
        Source/pch.hpp
)
