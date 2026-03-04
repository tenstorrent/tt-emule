add_library(matmul_kernels STATIC
    reader.cpp
    writer.cpp
    compute.cpp
)

target_include_directories(matmul_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(matmul_kernels PRIVATE tt_emule_lib)

add_executable(test_matmul test_matmul.cpp)

target_include_directories(test_matmul PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_matmul PRIVATE
    tt_emule_lib
    matmul_kernels
    Threads::Threads
)

add_test(NAME matmul COMMAND test_matmul)
