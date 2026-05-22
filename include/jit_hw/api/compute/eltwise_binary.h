// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT compute API — element-wise binary operations.
// add_tiles / sub_tiles / mul_tiles are defined in common.h.
// binary_op_init_common and binary_dest_reuse_tiles{,_init} are also in common.h.
// This header adds the per-op *_init stubs (no-ops in emulation).
#include "api/compute/common.h"

// Note: add_int_sfpu.h / sub_int_sfpu.h / mul_int_sfpu.h are pulled in
// transitively via common.h (see bottom of common.h).

namespace ckernel {

// Per-op init stubs.  On real hardware these reconfigure the math/unpack
// pipelines; in emulation they are no-ops.
ALWI void add_tiles_init(uint32_t /*icb0*/ = 0, uint32_t /*icb1*/ = 0,
                         bool /*acc_to_dest*/ = false) {}

ALWI void sub_tiles_init(uint32_t /*icb0*/ = 0, uint32_t /*icb1*/ = 0,
                         bool /*acc_to_dest*/ = false) {}

ALWI void mul_tiles_init(uint32_t /*icb0*/ = 0, uint32_t /*icb1*/ = 0,
                         bool /*acc_to_dest*/ = false) {}

ALWI void mul_tiles_init(uint32_t /*icb0*/, uint32_t /*icb1*/, uint32_t /*acc_to_dest*/) {}

} // namespace ckernel
