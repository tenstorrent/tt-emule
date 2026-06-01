// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for lgamma SFPU tile ops.
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/lgamma.h
// LLK family: llk_math_eltwise_unary_sfpu_lgamma (and binary/ternary fused variants)

#include <cmath>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void lgamma_stirling_tile_init() {}

// Computes lgamma(x) for x >= 0.5 and lgamma(1 - x) for x < 0.5 using the
// Stirling approximation. The reflection-formula correction for inputs < 0.5
// is applied separately by lgamma_adjusted_tile.
ALWI void lgamma_stirling_tile(uint32_t idst) {
    __emule_dst_check(idst, "lgamma_stirling_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        float z = (x < 0.5f) ? (1.0f - x) : x;
        __emule_dst[idst][i] = std::lgamma(z);
    }
}

ALWI void lgamma_stirling_float_tile_init() {}

// Float32-input variant. idst0 holds x, idst1 holds log(z) where
// z = (x < 0.5) ? (1 - x) : x. Writes lgamma(z) into idst2.
ALWI void lgamma_stirling_float_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2) {
    __emule_dst_check(idst0, "lgamma_stirling_float_tile");
    __emule_dst_check(idst1, "lgamma_stirling_float_tile");
    __emule_dst_check(idst2, "lgamma_stirling_float_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst0][i];
        float z = (x < 0.5f) ? (1.0f - x) : x;
        __emule_dst[idst2][i] = std::lgamma(z);
    }
}

ALWI void lgamma_adjusted_tile_init() {}

// Combines the Stirling-based lgamma approximation with the reflection-formula
// correction for inputs x < 0.5 and writes the final lgamma(x) value into idst3.
// idst0: lgamma_stirling result (lgamma(z))
// idst1: log|sin(pi * frac(x))| with integer adjustments (reflection correction term)
// idst2: original input x
// idst3: output lgamma(x) for -inf < x < inf
ALWI void lgamma_adjusted_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t idst3) {
    __emule_dst_check(idst0, "lgamma_adjusted_tile");
    __emule_dst_check(idst1, "lgamma_adjusted_tile");
    __emule_dst_check(idst2, "lgamma_adjusted_tile");
    __emule_dst_check(idst3, "lgamma_adjusted_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst2][i];
        __emule_dst[idst3][i] = std::lgamma(x);
    }
}

}  // namespace ckernel
