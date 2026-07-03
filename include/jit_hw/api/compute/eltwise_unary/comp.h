// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Unary comparison SFPU stubs for emulation.
// Each function compares DST elements against a scalar parameter or zero.
#include "jit_hw/api/compute/common.h"

namespace ckernel {

// ---- Unary comparisons: DST[idst][i] op param0 → 1 or 0 ----

// eq
ALWI void unary_eq_tile_init() {}

ALWI void unary_eq_tile(uint32_t idst, uint32_t param0) {
    float val;
    std::memcpy(&val, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] == val) ? 1.0f : 0.0f;
}

ALWI void unary_eq_tile_int32(uint32_t idst, uint32_t param0) {
    int32_t val = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x == val ? 1 : 0);
    }
}

// ne
ALWI void unary_ne_tile_init() {}

ALWI void unary_ne_tile(uint32_t idst, uint32_t param0) {
    float val;
    std::memcpy(&val, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] != val) ? 1.0f : 0.0f;
}

ALWI void unary_ne_tile_int32(uint32_t idst, uint32_t param0) {
    int32_t val = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x != val ? 1 : 0);
    }
}

// gt
ALWI void unary_gt_tile_init() {}

ALWI void unary_gt_tile(uint32_t idst, uint32_t param0) {
    float val;
    std::memcpy(&val, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] > val) ? 1.0f : 0.0f;
}

ALWI void unary_gt_tile_int32(uint32_t idst, uint32_t param0) {
    int32_t val = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x > val ? 1 : 0);
    }
}

// ge
ALWI void unary_ge_tile_init() {}

ALWI void unary_ge_tile(uint32_t idst, uint32_t param0) {
    float val;
    std::memcpy(&val, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] >= val) ? 1.0f : 0.0f;
}

ALWI void unary_ge_tile_int32(uint32_t idst, uint32_t param0) {
    int32_t val = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x >= val ? 1 : 0);
    }
}

// lt
ALWI void unary_lt_tile_init() {}

ALWI void unary_lt_tile(uint32_t idst, uint32_t param0) {
    float val;
    std::memcpy(&val, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] < val) ? 1.0f : 0.0f;
}

ALWI void unary_lt_tile_int32(uint32_t idst, uint32_t param0) {
    int32_t val = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x < val ? 1 : 0);
    }
}

// le
ALWI void unary_le_tile_init() {}

ALWI void unary_le_tile(uint32_t idst, uint32_t param0) {
    float val;
    std::memcpy(&val, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] <= val) ? 1.0f : 0.0f;
}

ALWI void unary_le_tile_int32(uint32_t idst, uint32_t param0) {
    int32_t val = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x <= val ? 1 : 0);
    }
}

// ---- Zero-comparison: DST[idst][i] op 0 → 1 or 0 ----

// gtz (greater than zero)
ALWI void gtz_tile_init() {}

ALWI void gtz_tile(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] > 0.0f) ? 1.0f : 0.0f;
}

ALWI void gtz_tile_int32(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x > 0 ? 1 : 0);
    }
}

// ltz (less than zero)
ALWI void ltz_tile_init() {}

ALWI void ltz_tile(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] < 0.0f) ? 1.0f : 0.0f;
}

ALWI void ltz_tile_int32(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x < 0 ? 1 : 0);
    }
}

// gez (greater than or equal zero)
ALWI void gez_tile_init() {}

ALWI void gez_tile(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] >= 0.0f) ? 1.0f : 0.0f;
}

ALWI void gez_tile_int32(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x >= 0 ? 1 : 0);
    }
}

// lez (less than or equal zero)
ALWI void lez_tile_init() {}

ALWI void lez_tile(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] <= 0.0f) ? 1.0f : 0.0f;
}

ALWI void lez_tile_int32(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x <= 0 ? 1 : 0);
    }
}

// eqz (equal zero)
ALWI void eqz_tile_init() {}

ALWI void eqz_tile(uint32_t idst) {
    // Integer-format operand (e.g. a UInt16 index): silicon runs eqz in integer
    // mode and yields an integer 1/0, which packs bit-exact. A float 1.0f would
    // be truncated to 0 by the UInt16 packer. Float operands take the usual path.
    if (__emule_dst_is_int(idst)) {
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            int32_t x = __emule_dst_load_i32(idst, i);
            __emule_dst_store_i32(idst, i, x == 0 ? 1 : 0);
        }
        return;
    }
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] == 0.0f) ? 1.0f : 0.0f;
}

ALWI void eqz_tile_int32(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x == 0 ? 1 : 0);
    }
}

ALWI void eqz_tile_uint16(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, (x & 0xFFFF) == 0 ? 1 : 0);
    }
}

ALWI void eqz_tile_uint32(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x == 0 ? 1 : 0);
    }
}

// nez (not equal zero)
ALWI void nez_tile_init() {}

ALWI void nez_tile(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = (__emule_compute_ctx().dst[idst][i] != 0.0f) ? 1.0f : 0.0f;
}

ALWI void nez_tile_int32(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x != 0 ? 1 : 0);
    }
}

ALWI void nez_tile_uint16(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, (x & 0xFFFF) != 0 ? 1 : 0);
    }
}

ALWI void nez_tile_uint32(uint32_t idst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x != 0 ? 1 : 0);
    }
}

} // namespace ckernel
