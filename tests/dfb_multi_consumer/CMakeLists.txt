add_library(dfb_mc_kernels STATIC
    producer.cpp
    consumer.cpp
)

target_include_directories(dfb_mc_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(dfb_mc_kernels PRIVATE tt_emule_lib)

add_executable(test_dfb_mc test_dfb_mc.cpp)

target_include_directories(test_dfb_mc PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_dfb_mc PRIVATE
    tt_emule_lib
    dfb_mc_kernels
    Threads::Threads
)

add_test(NAME dfb_multi_consumer COMMAND test_dfb_mc)
