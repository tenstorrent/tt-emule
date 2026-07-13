// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/i1.h`. Mirrors upstream's
// two-region polynomial fit for modified Bessel I1:
//   Region A (|x| <= 10): rational p(t)/q(t) on t = x², order 6/7. out = x * p/q.
//   Region B (|x| > 10):  asymptotic out = sign(x) * exp(|x|)/sqrt(|x|) * P(1/|x|)
//                         with degree-5 minimax P.
// Coefficients verbatim from
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_i1.h.
// |x| clamped to 88.5 to avoid std::exp overflow.

#include <cmath>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void i1_tile_init() {}

ALWI void i1_tile(uint32_t idst) {
    __emule_dst_check(idst, "i1_tile");

    // Region A — rational p(t)/q(t) on t = x², for |x| <= 10 (lines 108-126).
    constexpr float c0 = 5.0000000000e-01f;
    constexpr float c1 = 5.6819390506e-02f;
    constexpr float c2 = 1.9247245509e-03f;
    constexpr float c3 = 2.8397364076e-05f;
    constexpr float c4 = 2.0916867527e-07f;
    constexpr float c5 = 7.7937084564e-10f;
    constexpr float c6 = 1.2293555930e-12f;

    constexpr float d0 =  1.0f;
    constexpr float d1 = -1.1361218989e-02f;
    constexpr float d2 =  6.1268139689e-05f;
    constexpr float d3 = -1.9771712800e-07f;
    constexpr float d4 =  3.8127551116e-10f;
    constexpr float d5 = -3.1218170410e-13f;
    constexpr float d6 = -3.0635529988e-16f;
    constexpr float d7 =  7.4301498523e-19f;

    // Region B — asymptotic minimax P(1/|x|), |x| > 10 (lines 62-69).
    constexpr float a0 =  3.9894228967e-01f;
    constexpr float a1 = -1.4960495444e-01f;
    constexpr float a2 = -4.6652925320e-02f;
    constexpr float a3 = -4.3674591560e-02f;
    constexpr float a4 = -1.9748322314e-02f;
    constexpr float a5 = -3.3467922914e-01f;

    constexpr float X_CLAMP = 88.5f;
    constexpr float REGION_SPLIT = 10.0f;

    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        if (x >  X_CLAMP) x =  X_CLAMP;
        if (x < -X_CLAMP) x = -X_CLAMP;
        float ax = std::fabs(x);

        float out;
        if (ax <= REGION_SPLIT) {
            float t = x * x;
            float p = c6;
            p = p * t + c5;
            p = p * t + c4;
            p = p * t + c3;
            p = p * t + c2;
            p = p * t + c1;
            p = p * t + c0;
            float q = d7;
            q = q * t + d6;
            q = q * t + d5;
            q = q * t + d4;
            q = q * t + d3;
            q = q * t + d2;
            q = q * t + d1;
            q = q * t + d0;
            out = x * (p / q);
        } else {
            float y = 1.0f / ax;
            float P = a5;
            P = P * y + a4;
            P = P * y + a3;
            P = P * y + a2;
            P = P * y + a1;
            P = P * y + a0;
            float mag = std::exp(ax) / std::sqrt(ax) * P;
            out = std::copysign(mag, x);
        }

        __emule_compute_ctx().dst[idst][i] = out;
    }
}

}  // namespace ckernel
