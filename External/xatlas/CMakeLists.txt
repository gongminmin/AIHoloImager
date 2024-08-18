UpdateExternalLib("xatlas" "https://github.com/jpcy/xatlas.git" "f700c7790aaa030e794b52ba7791a05c085faf0c")

add_library(xatlas STATIC
    xatlas/source/xatlas/xatlas.cpp
    xatlas/source/xatlas/xatlas.h
)

target_include_directories(xatlas
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/xatlas/source
)

set_target_properties(xatlas PROPERTIES FOLDER "External")
