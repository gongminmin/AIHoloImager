add_executable(AIHoloImager
    Main.cpp
)

target_link_libraries(AIHoloImager
    PRIVATE
        cxxopts
        AIHoloImagerLib
)
