cmake_minimum_required(VERSION 3.16)
project(harmonic_compression)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling parallel processing.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include(FetchContent)

# LibE57Format
FetchContent_Declare(
        LibE57Format
        GIT_REPOSITORY https://github.com/asmaloney/LibE57Format.git
        GIT_TAG        v3.2.0
)

FetchContent_MakeAvailable(LibE57Format)

# dlib
FetchContent_Declare(
        dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v19.24
)

FetchContent_MakeAvailable(dlib)

add_executable(app main.cpp)

target_link_libraries(app PRIVATE
        E57Format
        dlib::dlib
        OpenMP::OpenMP_CXX
)
