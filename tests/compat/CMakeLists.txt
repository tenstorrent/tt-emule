add_executable(test_tt_metal_compat test_tt_metal_compat.cpp)

target_include_directories(test_tt_metal_compat PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Reuse the eltwise_add kernel objects — same kernel source files, no changes needed.
target_link_libraries(test_tt_metal_compat PRIVATE
    tt_emule_lib
    eltwise_add_kernels
    Threads::Threads
)

add_test(NAME tt_metal_compat COMMAND test_tt_metal_compat)
