// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// INT32 floor/trunc division SFPU stubs.
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void div_int32_floor_tile_init() {}
ALWI void div_int32_trunc_tile_init() {}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void div_int32_floor_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        // Floor division: round toward negative infinity
        if (b == 0) { __emule_dst_store_i32(odst, i, 0); continue; }
        int32_t q = a / b;
        if ((a ^ b) < 0 && q * b != a) q--;
        __emule_dst_store_i32(odst, i, q);
    }
}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void div_int32_trunc_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, b != 0 ? a / b : 0);
    }
}

} // namespace ckernel
