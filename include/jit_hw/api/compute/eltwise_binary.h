// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT compute API — element-wise binary operations.
// add_tiles / sub_tiles / mul_tiles are defined in common.h.
// binary_op_init_common and binary_dest_reuse_tiles{,_init} are also in common.h.
// This header adds the per-op *_init stubs (no-ops in emulation).
#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "jit_hw/llk_math_eltwise_binary.h"  // llk_math_eltwise_binary_init
#endif

// Note: add_int_sfpu.h / sub_int_sfpu.h / mul_int_sfpu.h are pulled in
// transitively via common.h (see bottom of common.h).

namespace ckernel {

// Per-op init stubs.  On real hardware these reconfigure the math/unpack
// pipelines; in emulation the only observable effect is the accumulate-into-DST
// mode (acc_to_dest), which the following *_tiles calls honour (see common.h).
ALWI void add_tiles_init(uint32_t /*icb0*/ = 0, uint32_t /*icb1*/ = 0,
                         bool acc_to_dest = false) { __emule_dest_accum_en = acc_to_dest; }

ALWI void sub_tiles_init(uint32_t /*icb0*/ = 0, uint32_t /*icb1*/ = 0,
                         bool acc_to_dest = false) { __emule_dest_accum_en = acc_to_dest; }

// Silicon's 2-arg mul_tiles_init sets acc_to_dest=true: on WH/BH "accumulation
// is default behaviour" for mul (tt_metal/hw/inc/api/compute/eltwise_binary.h),
// so the accumulate-into-DST idiom (e.g. groupnorm's `mul_tiles(x, ones)` summed
// into one dst, then reduced) works. DST is zeroed by tile_regs_acquire, so a
// lone mul still behaves like an overwrite. The 3-arg form honours the caller's
// explicit flag (e.g. Quasar control / explicit acc_to_dest=false).
ALWI void mul_tiles_init(uint32_t /*icb0*/ = 0, uint32_t /*icb1*/ = 0,
                         bool acc_to_dest = true) { __emule_dest_accum_en = acc_to_dest; }

// Silicon's overload (eltwise_binary.h:111) has a trailing call_line arg; emule ignores it.
ALWI void mul_tiles_init(uint32_t /*icb0*/, uint32_t /*icb1*/, uint32_t acc_to_dest,
                         uint32_t /*call_line*/ = __builtin_LINE()) { __emule_dest_accum_en = (acc_to_dest != 0); }

} // namespace ckernel
