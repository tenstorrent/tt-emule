// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Unary binop-with-scalar stubs for emulation.
// Each function applies a binary operation between each DST element and a scalar.
// Float variants: param1 is a uint32_t bit-cast of a float.
// INT32 variants: param1 is an int32_t reinterpreted as uint32_t.
#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/vector_mode.h"

namespace ckernel {

enum { ADD_UNARY = 0, SUB_UNARY = 1, MUL_UNARY = 2, DIV_UNARY = 3, RSUB_UNARY = 4 };

ALWI void binop_with_scalar_tile_init() {}

// LLK form used by kernels that need the VectorMode argument (e.g. SDPA's
// softplus/logsigmoid path). scalar is an fp32 bit pattern; binop_mode selects
// the op; vector_mode gates which faces are written (see vector_mode.h).
template <bool APPROXIMATE, int binop_mode>
ALWI void llk_math_eltwise_unary_sfpu_binop_with_scalar(
    uint32_t dst_index, uint32_t scalar, VectorMode vector_mode = VectorMode::RC) {
    float val;
    std::memcpy(&val, &scalar, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        if (!__emule_vector_mode_active(i, vector_mode)) continue;
        float& e = __emule_compute_ctx().dst[dst_index][i];
        if constexpr (binop_mode == ADD_UNARY)       e += val;
        else if constexpr (binop_mode == SUB_UNARY)  e -= val;
        else if constexpr (binop_mode == MUL_UNARY)  e *= val;
        else if constexpr (binop_mode == DIV_UNARY)  e /= val;
        else if constexpr (binop_mode == RSUB_UNARY) e = val - e;
    }
}

// Float variants: param1 is bit-cast float
ALWI void add_unary_tile(uint32_t idst, uint32_t param1) {
    float val;
    std::memcpy(&val, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] += val;
}

ALWI void sub_unary_tile(uint32_t idst, uint32_t param1) {
    float val;
    std::memcpy(&val, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] -= val;
}

ALWI void mul_unary_tile(uint32_t idst, uint32_t param1) {
    float val;
    std::memcpy(&val, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] *= val;
}

ALWI void div_unary_tile(uint32_t idst, uint32_t param1) {
    float val;
    std::memcpy(&val, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] /= val;
}

ALWI void rsub_unary_tile(uint32_t idst, uint32_t param1) {
    float val;
    std::memcpy(&val, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = val - __emule_compute_ctx().dst[idst][i];
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
