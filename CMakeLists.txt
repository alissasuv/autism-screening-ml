cmake_minimum_required(VERSION 3.16)

project(autism_screening_ml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.3.1
)

FetchContent_MakeAvailable(cxxopts)

add_library(csv_reader
    src/csv_reader.cpp
    src/feature_type.cpp
    src/statistics.cpp
    src/preprocessing.cpp
    src/binary_io.cpp
    src/dataset.cpp
)

target_include_directories(csv_reader PUBLIC include)

target_compile_options(csv_reader PRIVATE -Wall -Wextra)

add_executable(autism_screening
    app/main.cpp
    src/cli.cpp
)

target_link_libraries(autism_screening
    PRIVATE
    csv_reader
    cxxopts::cxxopts
)

target_compile_options(autism_screening PRIVATE -Wall -Wextra)