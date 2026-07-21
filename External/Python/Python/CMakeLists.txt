# Copyright (c) 2026 Minmin Gong
#

set(prebuilt_python_url "https://www.python.org/ftp/python/3.13.9/python-3.13.9-test-amd64.zip")
cmake_path(GET prebuilt_python_url FILENAME prebuilt_python_name)

set(prebuilt_python_path "${CMAKE_CURRENT_SOURCE_DIR}/${prebuilt_python_name}")
DownloadFile(${prebuilt_python_url} ${prebuilt_python_path})

set(prebuilt_python_dir "${CMAKE_CURRENT_SOURCE_DIR}/Python")
if(NOT EXISTS ${prebuilt_python_dir})
    message(STATUS "Extracting prebuilt python package to ${prebuilt_python_dir}")
    file(ARCHIVE_EXTRACT INPUT "${prebuilt_python_path}"
        DESTINATION "${prebuilt_python_dir}"
    )
endif()

set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_REGISTRY NEVER)
set(Python3_FIND_FRAMEWORK NEVER)
set(Python3_ROOT_DIR ${prebuilt_python_dir})
find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED GLOBAL)
set(Python3_FOUND ${Python3_FOUND} CACHE INTERNAL "" FORCE)
set(Python3_Interpreter_FOUND ${Python3_Interpreter_FOUND} CACHE INTERNAL "" FORCE)
set(Python3_EXECUTABLE ${Python3_EXECUTABLE} CACHE INTERNAL "" FORCE)
set(Python3_VERSION_MAJOR ${Python3_VERSION_MAJOR} CACHE INTERNAL "" FORCE)
set(Python3_VERSION_MINOR ${Python3_VERSION_MINOR} CACHE INTERNAL "" FORCE)
set(Python3_VERSION_PATCH ${Python3_VERSION_PATCH} CACHE INTERNAL "" FORCE)
set(Python3_VERSION ${Python3_VERSION} CACHE INTERNAL "" FORCE)
set(Python3_STDLIB ${Python3_STDLIB} CACHE INTERNAL "" FORCE)
set(Python3_RUNTIME_LIBRARY_DIRS ${Python3_RUNTIME_LIBRARY_DIRS} CACHE INTERNAL "" FORCE)

set(PythonInterp_FOUND ${Python3_Interpreter_FOUND} CACHE INTERNAL "" FORCE)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE INTERNAL "" FORCE)
set(PYTHON_VERSION_MAJOR ${Python3_VERSION_MAJOR} CACHE INTERNAL "" FORCE)
set(PYTHON_VERSION_MINOR ${Python3_VERSION_MINOR} CACHE INTERNAL "" FORCE)
set(PYTHON_VERSION_PATCH ${Python3_VERSION_PATCH} CACHE INTERNAL "" FORCE)
set(PYTHON_VERSION_STRING ${Python3_VERSION} CACHE INTERNAL "" FORCE)
