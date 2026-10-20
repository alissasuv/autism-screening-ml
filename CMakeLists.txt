cmake_minimum_required(VERSION 3.16)

project(autism_screening_ml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(csv_reader
    src/csv_reader.cpp
    src/feature_type.cpp
)

target_include_directories(csv_reader PUBLIC include)
target_compile_options(csv_reader PRIVATE -Wall -Wextra)

add_executable(autism_screening app/main.cpp)

target_link_libraries(autism_screening PRIVATE csv_reader)
target_compile_options(autism_screening PRIVATE -Wall -Wextra)
