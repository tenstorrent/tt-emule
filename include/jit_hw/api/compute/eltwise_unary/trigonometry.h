// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stubs for trigonometric SFPU tile ops.
// All ops operate on DST (float32 storage) element-wise via std:: math.

#include <cmath>
#include <cstdint>

namespace ckernel {

ALWI void sin_tile_init() {}
ALWI void sin_tile(uint32_t idst) {
    __emule_dst_check(idst, "sin_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::sin(__emule_dst[idst][i]);
}

ALWI void cos_tile_init() {}
ALWI void cos_tile(uint32_t idst) {
    __emule_dst_check(idst, "cos_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::cos(__emule_dst[idst][i]);
}

ALWI void tan_tile_init() {}
ALWI void tan_tile(uint32_t idst) {
    __emule_dst_check(idst, "tan_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::tan(__emule_dst[idst][i]);
}

// Templated on fast_and_approx to match upstream compute_kernel_api.h; the
// kernel SFPU_OP_CHAIN calls tanh_tile<0u>(0) / tanh_tile_init<0u>().
template <bool fast_and_approx = false>
ALWI void tanh_tile_init() {}
template <bool fast_and_approx = false>
ALWI void tanh_tile(uint32_t idst) {
    __emule_dst_check(idst, "tanh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::tanh(__emule_dst[idst][i]);
}

ALWI void asin_tile_init() {}
ALWI void asin_tile(uint32_t idst) {
    __emule_dst_check(idst, "asin_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::asin(__emule_dst[idst][i]);
}

ALWI void acos_tile_init() {}
ALWI void acos_tile(uint32_t idst) {
    __emule_dst_check(idst, "acos_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::acos(__emule_dst[idst][i]);
}

ALWI void atan_tile_init() {}
ALWI void atan_tile(uint32_t idst) {
    __emule_dst_check(idst, "atan_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::atan(__emule_dst[idst][i]);
}

ALWI void acosh_tile_init() {}
ALWI void acosh_tile(uint32_t idst) {
    __emule_dst_check(idst, "acosh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::acosh(__emule_dst[idst][i]);
}

ALWI void asinh_tile_init() {}
ALWI void asinh_tile(uint32_t idst) {
    __emule_dst_check(idst, "asinh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::asinh(__emule_dst[idst][i]);
}

ALWI void atanh_tile_init() {}
ALWI void atanh_tile(uint32_t idst) {
    __emule_dst_check(idst, "atanh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::atanh(__emule_dst[idst][i]);
}

ALWI void cosh_tile_init() {}
ALWI void cosh_tile(uint32_t idst) {
    __emule_dst_check(idst, "cosh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::cosh(__emule_dst[idst][i]);
}

ALWI void sinh_tile_init() {}
ALWI void sinh_tile(uint32_t idst) {
    __emule_dst_check(idst, "sinh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::sinh(__emule_dst[idst][i]);
}

} // namespace ckernel
