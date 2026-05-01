add_library(dfb_passthrough_kernels STATIC
    producer.cpp
    consumer.cpp
)

target_include_directories(dfb_passthrough_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(dfb_passthrough_kernels PRIVATE tt_emule_lib)

add_executable(test_dfb_passthrough test_dfb_passthrough.cpp)

target_include_directories(test_dfb_passthrough PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_dfb_passthrough PRIVATE
    tt_emule_lib
    dfb_passthrough_kernels
    Threads::Threads
)

add_test(NAME dfb_passthrough COMMAND test_dfb_passthrough)
