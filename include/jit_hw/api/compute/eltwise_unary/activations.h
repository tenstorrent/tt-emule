// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stubs for activation / absolute-value tile operations.
// DST is __emule_dst[][] (float array); operations are host-side math.

#include <cmath>
#include <cstdint>

namespace ckernel {

ALWI void abs_tile_init() {}

ALWI void abs_tile(uint32_t idst) {
    __emule_dst_check(idst, "abs_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::fabs(__emule_dst[idst][i]);
}

ALWI void abs_tile_int32(uint32_t idst) {
    __emule_dst_check(idst, "abs_tile_int32");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        if (v < 0) v = -v;
        __emule_dst_store_i32(idst, i, v);
    }
}

} // namespace ckernel
