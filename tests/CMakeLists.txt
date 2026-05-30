# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(tilize)
add_subdirectory(dram)

# Standalone host-side validation for the Bfp8_b codec in
# include/jit_hw/api/compute/bfp8.h. Header-only — no link against tt_emule_lib.
add_executable(test_bfp8_roundtrip bfp8_roundtrip_test.cpp)
target_include_directories(test_bfp8_roundtrip PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
add_test(NAME bfp8_roundtrip COMMAND test_bfp8_roundtrip)
