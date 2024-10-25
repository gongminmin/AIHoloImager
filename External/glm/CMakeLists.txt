UpdateExternalLib("glm" "https://github.com/g-truc/glm.git" "33b4a621a697a305bc3a7610d290677b96beb181")

set(BUILD_SHARED_LIBS OFF)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLM_ENABLE_CXX_20 ON CACHE BOOL "" FORCE)

add_subdirectory(glm EXCLUDE_FROM_ALL)

set_target_properties(glm PROPERTIES FOLDER "External")
