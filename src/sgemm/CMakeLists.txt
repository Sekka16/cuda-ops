add_library(sgemm sgemm.cu)
target_include_directories(sgemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})