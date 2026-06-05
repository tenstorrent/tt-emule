// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stubs for ReLU SFPU tile ops.

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace ckernel {

ALWI void relu_tile_init() {}

ALWI void relu_tile(uint32_t idst) {
    __emule_dst_check(idst, "relu_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::max(0.0f, __emule_dst[idst][i]);
}

ALWI void relu_tile_int32(uint32_t idst) {
    __emule_dst_check(idst, "relu_tile_int32");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, std::max<int32_t>(0, v));
    }
}

ALWI void relu_max_tile_init() {}

// max(0, min(v, threshold))  — threshold passed as the raw FP32 bit pattern in
// a uint32 (memcpy'd back to float), matching the other parameterized ops here
// (e.g. clamped_silu, leaky_relu_tile).
ALWI void relu_max_tile(uint32_t idst, uint32_t threshold) {
    __emule_dst_check(idst, "relu_max_tile");
    float t;
    std::memcpy(&t, &threshold, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float v = __emule_dst[idst][i];
        if (v < 0.0f) v = 0.0f;
        if (v > t) v = t;
        __emule_dst[idst][i] = v;
    }
}

ALWI void relu_min_tile_init() {}

// max(threshold, v)
ALWI void relu_min_tile(uint32_t idst, uint32_t threshold) {
    __emule_dst_check(idst, "relu_min_tile");
    float t;
    std::memcpy(&t, &threshold, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float v = __emule_dst[idst][i];
        __emule_dst[idst][i] = (v > t) ? v : t;
    }
}

ALWI void leaky_relu_tile_init() {}

ALWI void leaky_relu_tile(uint32_t idst, uint32_t slope_bits) {
    __emule_dst_check(idst, "leaky_relu_tile");
    float slope;
    std::memcpy(&slope, &slope_bits, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float v = __emule_dst[idst][i];
        __emule_dst[idst][i] = (v > 0.0f) ? v : slope * v;
    }
}

} // namespace ckernel
