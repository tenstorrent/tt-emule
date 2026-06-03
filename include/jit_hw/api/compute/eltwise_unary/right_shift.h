// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/right_shift.h`.
// Arithmetic right shift on int32 DST by `param0` bits.
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_right_shift.h
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
