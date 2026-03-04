add_library(eltwise_add_kernels STATIC
    reader.cpp
    writer.cpp
    compute.cpp
)

target_include_directories(eltwise_add_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(eltwise_add_kernels PRIVATE tt_emule_lib)

add_executable(test_eltwise_add test_eltwise_add.cpp)

target_include_directories(test_eltwise_add PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_eltwise_add PRIVATE
    tt_emule_lib
    eltwise_add_kernels
    Threads::Threads
)

add_test(NAME eltwise_add COMMAND test_eltwise_add)
