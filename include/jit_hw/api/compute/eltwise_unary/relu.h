// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
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

// relu_max: clamp(x, 0, cap). Real LLK: ckernel_sfpu_relu.h relu_max.
ALWI void relu_max_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "relu_max_tile");
    float cap;
    std::memcpy(&cap, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        if (x > cap) x = cap;
        if (x < 0.0f) x = 0.0f;
        __emule_dst[idst][i] = x;
    }
}

// Pack-path variant: same math as relu_max_tile.
ALWI void relu_max_tile_pack(uint32_t idst, uint32_t param0) {
    relu_max_tile(idst, param0);
}

ALWI void relu_max_tile_int32(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "relu_max_tile_int32");
    int32_t cap = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        if (v > cap) v = cap;
        if (v < 0) v = 0;
        __emule_dst_store_i32(idst, i, v);
    }
}

ALWI void relu_max_tile_init() {}
ALWI void relu_max_tile_init_pack() {}

// relu_min: max(x, thr). Real LLK: ckernel_sfpu_relu.h relu_min (clamps up to threshold).
ALWI void relu_min_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "relu_min_tile");
    float thr;
    std::memcpy(&thr, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::max(thr, __emule_dst[idst][i]);
}

ALWI void relu_min_tile_int32(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "relu_min_tile_int32");
    int32_t thr = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, std::max(thr, v));
    }
}

ALWI void relu_min_tile_init() {}

// leaky_relu: x > 0 ? x : slope*x.
ALWI void leaky_relu_tile(uint32_t idst, uint32_t slope = 0) {
    __emule_dst_check(idst, "leaky_relu_tile");
    float s;
    std::memcpy(&s, &slope, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = x > 0.0f ? x : s * x;
    }
}

ALWI void leaky_relu_tile_init() {}

} // namespace ckernel
