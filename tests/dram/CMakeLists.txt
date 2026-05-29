# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# DRAM model regression coverage. These tests directly exercise the
# tt_emule::Core DRAM role and the address translation invariants that
# the host bridges (in tt-metal/.../emulated_program_runner.cpp)
# rely on.

add_executable(test_dram_model test_dram_model.cpp)

target_include_directories(test_dram_model PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_dram_model PRIVATE
    tt_emule_lib
    Threads::Threads
)

add_test(NAME dram_model COMMAND test_dram_model)
