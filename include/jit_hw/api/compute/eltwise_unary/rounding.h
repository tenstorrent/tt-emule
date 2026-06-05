// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stubs for rounding SFPU tile ops.
// trunc, ceil, floor operate element-wise on DST float32 storage; frac = x - trunc(x).

#include <cmath>
#include <cstdint>

namespace ckernel {

ALWI void rounding_op_tile_init() {}

ALWI void round_tile_init() {}

// Round each element to `decimals` decimal places (round-to-nearest-even).
// Real LLK: _calculate_round_ = (1/10^d) * round_even(x * 10^d).
ALWI void round_tile(uint32_t idst, int32_t decimals) {
    __emule_dst_check(idst, "round_tile");
    float scale = std::pow(10.0f, (float)decimals);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = std::nearbyint(x * scale) / scale;
    }
}

ALWI void trunc_tile(uint32_t idst) {
    __emule_dst_check(idst, "trunc_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::trunc(__emule_dst[idst][i]);
}

ALWI void ceil_tile(uint32_t idst) {
    __emule_dst_check(idst, "ceil_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::ceil(__emule_dst[idst][i]);
}

ALWI void floor_tile(uint32_t idst) {
    __emule_dst_check(idst, "floor_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::floor(__emule_dst[idst][i]);
}

ALWI void frac_tile(uint32_t idst) {
    __emule_dst_check(idst, "frac_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = x - std::trunc(x);
    }
}

} // namespace ckernel
