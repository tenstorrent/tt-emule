// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Binary shift SFPU stubs. Silicon (ckernel_sfpu_shift.h) clamps the per-element
// shift amount: amount < 0 || amount >= 32 -> result 0 (NOT a C `b & 31` mask).
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void binary_shift_tile_init() {}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void binary_left_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, (b < 0 || b >= 32) ? 0 : (a << b));
    }
}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void binary_right_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, (b < 0 || b >= 32) ? 0 : (a >> b));  // arithmetic
    }
}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void binary_logical_right_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        uint32_t a;
        std::memcpy(&a, &__emule_dst[idst0][i], sizeof(uint32_t));
        int32_t b = __emule_dst_load_i32(idst1, i);
        uint32_t r = (b < 0 || b >= 32) ? 0u : (a >> b);  // zero-fill
        std::memcpy(&__emule_dst[odst][i], &r, sizeof(uint32_t));
    }
}

} // namespace ckernel
