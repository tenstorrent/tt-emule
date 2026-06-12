# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Per-RISC circular-buffer pointer regression (issue #139). Exercises the
# jit_hw CB pointer helpers (emule_cb_ptr.h / cb_interface.h) directly: per-RISC
# isolation (the race fix), get_local_cb_interface write-back, and ring wrap.

add_executable(test_cb_ptr test_cb_ptr.cpp)

target_include_directories(test_cb_ptr PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/jit_hw   # for the tt-metalium/ constants mirror
)

# NUM_CIRCULAR_BUFFERS is arch-gated in circular_buffer_constants.h (32 on WH,
# 64 otherwise). The test is arch-independent (it only needs NUM > kCb), so leave
# the arch define unset to build against the larger 64-CB ring (BH/Quasar sizing).

target_link_libraries(test_cb_ptr PRIVATE
    tt_emule_lib
    Threads::Threads
)

add_test(NAME cb_ptr COMMAND test_cb_ptr)
