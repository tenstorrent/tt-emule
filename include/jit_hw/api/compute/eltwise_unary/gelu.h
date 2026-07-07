// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stub for GELU SFPU tile op.
// Implements the exact GELU formula: 0.5 * x * (1 + erf(x / sqrt(2))).

#include <cmath>
#include <cstdint>

namespace ckernel {

template <bool fast_and_approx = true>
ALWI void gelu_tile_init() {}

template <bool fast_and_approx = true>
ALWI void gelu_tile(uint32_t idst) {
    __emule_dst_check(idst, "gelu_tile");
    static const float kInvSqrt2 = 1.0f / std::sqrt(2.0f);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        __emule_compute_ctx().dst[idst][i] = 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
    }
}

template <bool fast_and_approx = true>
ALWI void gelu_tile_init_pack() {
    gelu_tile_init<fast_and_approx>();
}

template <bool fast_and_approx = true>
ALWI void gelu_tile_pack(uint32_t idst) {
    gelu_tile<fast_and_approx>(idst);
}

ALWI void gelu_tanh_tile_init() {}

ALWI void gelu_tanh_tile(uint32_t idst) {
    __emule_dst_check(idst, "gelu_tanh_tile");
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        float cubic = x * x * x;
        __emule_compute_ctx().dst[idst][i] = 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * cubic)));
    }
}

ALWI void gelu_tanh_tile_init_pack() {
    gelu_tanh_tile_init();
}

ALWI void gelu_tanh_tile_pack(uint32_t idst) {
    gelu_tanh_tile(idst);
}

// gelu'(x) = Phi(x) + x*phi(x), exact (matches gelu_tile's erf form).
// phi(x) = exp(-x^2/2)/sqrt(2*pi); kInvSqrt2Pi = 1/sqrt(2*pi).
template <bool fast_and_approx = false>
ALWI void gelu_derivative_tile_init() {}

template <bool fast_and_approx = false>
ALWI void gelu_derivative_tile(uint32_t idst) {
    __emule_dst_check(idst, "gelu_derivative_tile");
    static const float kInvSqrt2 = 1.0f / std::sqrt(2.0f);
    static const float kInvSqrt2Pi = 0.3989422804014327f;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        float cdf = 0.5f * (1.0f + std::erf(x * kInvSqrt2));
        float pdf = kInvSqrt2Pi * std::exp(-0.5f * x * x);
        __emule_compute_ctx().dst[idst][i] = cdf + x * pdf;
    }
}

} // namespace ckernel
