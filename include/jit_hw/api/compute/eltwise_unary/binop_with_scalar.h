// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Unary binop-with-scalar stubs for emulation.
// Each function applies a binary operation between each DST element and a scalar.
// Float variants: param1 is a uint32_t bit-cast of a float.
// INT32 variants: param1 is an int32_t reinterpreted as uint32_t.
#include "jit_hw/api/compute/common.h"

namespace ckernel {

enum { ADD_UNARY = 0, SUB_UNARY = 1, MUL_UNARY = 2, DIV_UNARY = 3, RSUB_UNARY = 4 };

ALWI void binop_with_scalar_tile_init() {}

// Float variants: param1 is bit-cast float
ALWI void add_unary_tile(uint32_t idst, uint32_t param1) {
    float val;
    std::memcpy(&val, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] += val;
}

ALWI void sub_unary_tile(uint32_t idst, uint32_t param1) {
    float val;
    std::memcpy(&val, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] -= val;
}

ALWI void mul_unary_tile(uint32_t idst, uint32_t param1) {
    float val;
    std::memcpy(&val, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] *= val;
}

ALWI void div_unary_tile(uint32_t idst, uint32_t param1) {
    float val;
    std::memcpy(&val, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] /= val;
}

ALWI void rsub_unary_tile(uint32_t idst, uint32_t param1) {
    float val;
    std::memcpy(&val, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = val - __emule_dst[idst][i];
}

// INT32 variants: param1 is int32_t reinterpreted as uint32_t
ALWI void add_unary_tile_int32(uint32_t idst, uint32_t param1) {
    int32_t val = static_cast<int32_t>(param1);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x + val);
    }
}

ALWI void sub_unary_tile_int32(uint32_t idst, uint32_t param1) {
    int32_t val = static_cast<int32_t>(param1);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t x = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, x - val);
    }
}

} // namespace ckernel
