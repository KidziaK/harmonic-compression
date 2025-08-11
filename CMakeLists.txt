cmake_minimum_required(VERSION 3.22)
project(harmonic_compression)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling parallel processing.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include(FetchContent)

FetchContent_Declare(
        LibE57Format
        GIT_REPOSITORY https://github.com/asmaloney/LibE57Format.git
        GIT_TAG        v3.2.0
)
FetchContent_MakeAvailable(LibE57Format)

FetchContent_Declare(
        dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v19.24
)
FetchContent_MakeAvailable(dlib)

add_library(compression_lib compression_lib.cpp)

target_link_libraries(compression_lib PUBLIC
        E57Format
        dlib::dlib
        OpenMP::OpenMP_CXX
)

target_include_directories(compression_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(app main.cpp)

FetchContent_Declare(
        nanoflann
        GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
        GIT_TAG        v1.5.5
)
FetchContent_MakeAvailable(nanoflann)

target_link_libraries(compression_lib PUBLIC
        E57Format
        dlib::dlib
        OpenMP::OpenMP_CXX
        nanoflann
)

add_executable(run_tests test_runner.cpp)
target_link_libraries(run_tests PRIVATE compression_lib)

add_test(NAME compression_correctness_tests
        COMMAND $<TARGET_FILE:run_tests>
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)