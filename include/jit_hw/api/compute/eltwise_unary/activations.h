#pragma once
// Emulator stubs for activation / absolute-value tile operations.
// DST is __emule_dst[][] (float array); operations are host-side math.

#include <algorithm>
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

// hardsigmoid: clamp((x+3)/6, 0, 1)
ALWI void hardsigmoid_tile_init() {}

ALWI void hardsigmoid_tile(uint32_t idst) {
    __emule_dst_check(idst, "hardsigmoid_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = std::clamp((x + 3.0f) / 6.0f, 0.0f, 1.0f);
    }
}

// softsign: x / (1 + |x|)
ALWI void softsign_tile_init() {}

ALWI void softsign_tile(uint32_t idst) {
    __emule_dst_check(idst, "softsign_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = x / (1.0f + std::fabs(x));
    }
}

} // namespace ckernel
