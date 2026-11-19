cmake_minimum_required(VERSION 3.16)

project(autism_screening_ml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)

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
    src/decision_stump.cpp
    src/split_finder.cpp
    src/cart.cpp
    src/random_forest.cpp
    src/lab5_data.cpp
)

target_include_directories(csv_reader PUBLIC include)

target_link_libraries(csv_reader PUBLIC Threads::Threads)

target_compile_options(csv_reader PRIVATE
    -Wall
    -Wextra
)

add_executable(autism_screening
    app/main.cpp
    src/cli.cpp
)

target_link_libraries(autism_screening PRIVATE
    csv_reader
    cxxopts::cxxopts
)

target_compile_options(autism_screening PRIVATE
    -Wall
    -Wextra
)

add_executable(autism_model_demo
    app/model_demo.cpp
)

target_link_libraries(autism_model_demo PRIVATE csv_reader)

target_compile_options(autism_model_demo PRIVATE
    -Wall
    -Wextra
)

# Automated tests
enable_testing()

add_executable(autism_tests
    tests/test_main.cpp
)

target_link_libraries(autism_tests PRIVATE csv_reader)

target_compile_options(autism_tests PRIVATE
    -Wall
    -Wextra
)

add_test(
    NAME autism_pipeline_tests
    COMMAND autism_tests
)
add_executable(decision_stump_tests
    tests/test_decision_stump.cpp
)

target_link_libraries(decision_stump_tests PRIVATE csv_reader)

target_compile_options(decision_stump_tests PRIVATE
    -Wall
    -Wextra
)

add_test(
    NAME decision_stump_tests
    COMMAND decision_stump_tests
)

option(ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
    OFF
)

if(ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        foreach(target csv_reader autism_screening autism_tests decision_stump_tests autism_model_demo)
            target_compile_options(${target} PRIVATE
                -fsanitize=address,undefined
                -fno-omit-frame-pointer
                -g
            )

            target_link_options(${target} PRIVATE
                -fsanitize=address,undefined
            )
        endforeach()
    else()
        message(FATAL_ERROR
            "Sanitizers require Clang or GCC"
        )
    endif()
endif()

add_executable(matrix_slice_tests
    tests/test_matrix_slice.cpp
)

target_link_libraries(matrix_slice_tests PRIVATE csv_reader)

target_compile_options(matrix_slice_tests PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(matrix_slice_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )
    target_link_options(matrix_slice_tests PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_test(
    NAME matrix_slice_tests
    COMMAND matrix_slice_tests
)

add_executable(cart_demo
    app/cart_demo.cpp
)

target_link_libraries(cart_demo PRIVATE csv_reader)

target_compile_options(cart_demo PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(cart_demo PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(cart_demo PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_executable(cart_tests
    tests/test_cart.cpp
)

target_link_libraries(cart_tests PRIVATE csv_reader)

target_compile_options(cart_tests PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(cart_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(cart_tests PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_test(
    NAME cart_tests
    COMMAND cart_tests
)


# Lab 4
add_executable(split_finder_tests
    tests/test_split_finder.cpp
)

target_link_libraries(split_finder_tests PRIVATE csv_reader)

target_compile_options(split_finder_tests PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(split_finder_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(split_finder_tests PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_test(
    NAME split_finder_tests
    COMMAND split_finder_tests
)

add_executable(cart_weighted_tests
    tests/test_cart_weighted.cpp
)

target_link_libraries(cart_weighted_tests PRIVATE csv_reader)

target_compile_options(cart_weighted_tests PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(cart_weighted_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(cart_weighted_tests PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_test(
    NAME cart_weighted_tests
    COMMAND cart_weighted_tests
)


# Lab 4 — Random Forest regression tests
add_executable(random_forest_regression_tests
    tests/test_random_forest_regression.cpp
)

target_link_libraries(
    random_forest_regression_tests
    PRIVATE csv_reader
)

target_compile_options(
    random_forest_regression_tests PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
        random_forest_regression_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(
        random_forest_regression_tests PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_test(
    NAME random_forest_regression_tests
    COMMAND random_forest_regression_tests
)

add_executable(random_forest_classification_tests
    tests/test_random_forest_classification.cpp
)

target_link_libraries(
    random_forest_classification_tests
    PRIVATE csv_reader
)

target_compile_options(
    random_forest_classification_tests PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
        random_forest_classification_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(
        random_forest_classification_tests PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_test(
    NAME random_forest_classification_tests
    COMMAND random_forest_classification_tests
)

add_executable(forest_parallel_tests
    tests/test_forest_parallel.cpp
)

target_link_libraries(
    forest_parallel_tests
    PRIVATE csv_reader
)

target_compile_options(forest_parallel_tests PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(forest_parallel_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(forest_parallel_tests PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_test(
    NAME forest_parallel_tests
    COMMAND forest_parallel_tests
)

add_executable(random_forest_demo
    app/random_forest_demo.cpp
)

target_link_libraries(
    random_forest_demo
    PRIVATE csv_reader
)

target_compile_options(random_forest_demo PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(random_forest_demo PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(random_forest_demo PRIVATE
        -fsanitize=address,undefined
    )
endif()

# Lab 5

add_executable(prepare_data
    app/prepare_data.cpp
)

target_link_libraries(prepare_data PRIVATE
    csv_reader
)

target_compile_options(prepare_data PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(prepare_data PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(prepare_data PRIVATE
        -fsanitize=address,undefined
    )
endif()

# Lab 5 - Data preparation tests

add_executable(lab5_data_tests
    tests/test_lab5_data.cpp
)

target_link_libraries(lab5_data_tests PRIVATE
    csv_reader
)

target_compile_options(lab5_data_tests PRIVATE
    -Wall
    -Wextra
)

add_test(
    NAME lab5_data_tests
    COMMAND lab5_data_tests
)
# Lab 5 - YAML-backed preprocessing adapter

find_package(yaml-cpp 0.8 CONFIG QUIET)

if(NOT TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        yaml_cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0
        GIT_SHALLOW TRUE
    )

    FetchContent_MakeAvailable(yaml_cpp)
endif()

target_sources(csv_reader PRIVATE
    src/lab5_preprocessing.cpp
)

target_link_libraries(csv_reader PUBLIC
    yaml-cpp::yaml-cpp
)

add_executable(lab5_preprocessing_tests
    tests/test_lab5_preprocessing.cpp
)

target_link_libraries(lab5_preprocessing_tests PRIVATE
    csv_reader
)

target_compile_options(lab5_preprocessing_tests PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(lab5_preprocessing_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(lab5_preprocessing_tests PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_test(
    NAME lab5_preprocessing_tests
    COMMAND lab5_preprocessing_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/configs/preprocessing.yaml"
)

# Lab 5 - Shared regression and classification metrics

target_sources(csv_reader PRIVATE
    src/metrics.cpp
)

add_executable(lab5_metrics_tests
    tests/test_lab5_metrics.cpp
)

target_link_libraries(lab5_metrics_tests PRIVATE
    csv_reader
)

target_compile_options(lab5_metrics_tests PRIVATE
    -Wall
    -Wextra
)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(lab5_metrics_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_link_options(lab5_metrics_tests PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_test(
    NAME lab5_metrics_tests
    COMMAND lab5_metrics_tests
)
