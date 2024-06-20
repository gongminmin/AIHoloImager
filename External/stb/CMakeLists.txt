UpdateExternalLib("stb" "https://github.com/nothings/stb.git" "b42009b3b9d4ca35bc703f5310eedc74f584be58")

add_library(stb INTERFACE)
target_include_directories(stb
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/stb
)
