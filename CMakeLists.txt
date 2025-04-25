cmake_minimum_required(VERSION 3.18)
project(cuda_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
enable_language(CUDA)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_subdirectory(src/reduce)
add_subdirectory(src/histogram)
add_subdirectory(src/sgemm)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

add_executable(test_reduce test/test_reduce.cpp)
target_link_libraries(test_reduce reduce)

add_executable(test_histogram test/test_histogram.cpp)
target_link_libraries(test_histogram histogram)

add_executable(test_sgemm test/test_sgemm.cpp)
target_link_libraries(test_sgemm sgemm)