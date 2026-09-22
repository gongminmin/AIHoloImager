add_executable(AIHoloImager
    Main.cpp
)

target_link_libraries(AIHoloImager
    PRIVATE
        cxxopts
        AIHoloImagerLib
)

set_target_properties(AIHoloImager PROPERTIES FOLDER "App")
