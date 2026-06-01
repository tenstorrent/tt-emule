// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/right_shift.h`. Intercepts the
// upstream include path which pulls in `ckernel_sfpu_right_shift.h` (an
// LLK-only header that references SFPU intrinsics).
//
// Semantics: each int32 element of DST tile `idst` is arithmetically
// right-shifted by `param0` bits. DST is signed int32 for this op, so we
// load/store via the int32 accessors and use C++ signed `>>` which is
// arithmetic on standard implementations.
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/eltwise_unary/right_shift.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_right_shift.h
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void right_shift_tile_init() {}

ALWI void right_shift_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "right_shift_tile");
    const uint32_t shift = param0 & 31u;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, v >> shift);
    }
}

}  // namespace ckernel
