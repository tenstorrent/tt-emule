// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for isinf/isnan/isfinite/isposinf/isneginf SFPU tile ops.
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/isinf_isnan.h
// LLK family: llk_math_eltwise_unary_sfpu_isinf_isnan

#include <cmath>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void isinf_tile_init() {}

ALWI void isinf_tile(uint32_t idst) {
    __emule_dst_check(idst, "isinf_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        __emule_compute_ctx().dst[idst][i] = std::isinf(x) ? 1.0f : 0.0f;
    }
}

ALWI void isnan_tile_init() {}

ALWI void isnan_tile(uint32_t idst) {
    __emule_dst_check(idst, "isnan_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        __emule_compute_ctx().dst[idst][i] = std::isnan(x) ? 1.0f : 0.0f;
    }
}

ALWI void isfinite_tile_init() {}

ALWI void isfinite_tile(uint32_t idst) {
    __emule_dst_check(idst, "isfinite_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        __emule_compute_ctx().dst[idst][i] = std::isfinite(x) ? 1.0f : 0.0f;
    }
}

ALWI void isposinf_tile_init() {}

ALWI void isposinf_tile(uint32_t idst) {
    __emule_dst_check(idst, "isposinf_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        __emule_compute_ctx().dst[idst][i] = (x > 0.0f && std::isinf(x)) ? 1.0f : 0.0f;
    }
}

ALWI void isneginf_tile_init() {}

ALWI void isneginf_tile(uint32_t idst) {
    __emule_dst_check(idst, "isneginf_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        __emule_compute_ctx().dst[idst][i] = (x < 0.0f && std::isinf(x)) ? 1.0f : 0.0f;
    }
}

}  // namespace ckernel
