# Negative tests demonstrating that ASan / bridge-function bounds checks
# fire on intentional out-of-bounds accesses. Built only when
# TT_EMULE_ASAN=ON because two of the three tests need the ASan runtime to
# observe a poisoned-memory write (the third — DRAM bounds via
# __emule_dram_ptr — uses an always-on bounds check and would also work in
# non-ASan builds, but it shares this gating for simplicity).
#
# All tests are expected to die. CTest accepts the run as a pass when the
# program exits non-zero AND the stderr contains a recognizable abort marker.

add_executable(test_asan_oob_slot_tail oob_slot_tail_test.cpp)
target_include_directories(test_asan_oob_slot_tail PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_asan_oob_slot_tail PRIVATE tt_emule_lib Threads::Threads)
add_test(NAME asan_oob_slot_tail COMMAND test_asan_oob_slot_tail)
set_tests_properties(asan_oob_slot_tail PROPERTIES
    LABELS "asan"
    WILL_FAIL TRUE)

add_executable(test_asan_oob_l1_alloc oob_l1_alloc_test.cpp)
target_include_directories(test_asan_oob_l1_alloc PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_asan_oob_l1_alloc PRIVATE tt_emule_lib Threads::Threads)
add_test(NAME asan_oob_l1_alloc COMMAND test_asan_oob_l1_alloc)
set_tests_properties(asan_oob_l1_alloc PROPERTIES
    LABELS "asan"
    WILL_FAIL TRUE)

add_executable(test_asan_oob_dram oob_dram_test.cpp)
target_include_directories(test_asan_oob_dram PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_asan_oob_dram PRIVATE tt_emule_lib Threads::Threads)
add_test(NAME asan_oob_dram COMMAND test_asan_oob_dram)
set_tests_properties(asan_oob_dram PROPERTIES
    LABELS "asan"
    WILL_FAIL TRUE)
