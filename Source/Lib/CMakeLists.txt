add_library(AIHoloImagerLib STATIC
    Include/AIHoloImager/AIHoloImager.hpp
    Include/AIHoloImager/Mesh.hpp
    Include/AIHoloImager/Texture.hpp
    Source/AIHoloImager.cpp
    Source/Mesh.cpp
    Source/Texture.cpp
)

target_include_directories(AIHoloImagerLib
    PUBLIC
        Include
)

target_link_libraries(AIHoloImagerLib
    PRIVATE
        assimp
        stb
)
