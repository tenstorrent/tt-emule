// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/softplus.h`. Intercepts the
// upstream include path which pulls in `ckernel_sfpu_softplus.h` (an LLK-only
// header that references SFPU intrinsics).
//
// Semantics (per element, mirrors upstream ckernel_sfpu_softplus.h):
//   t = beta * x
//   if (t > threshold): out = x   (pass-through)
//   else:
//     a = |t|
//     residual = degree-8 Horner poly of ln(1+exp(-a)) on [0, 5]
//     if (a > 5):
//       e = exp(-a)
//       residual = e * (1 + e*(-1/2 + e/3))   // 3-term Taylor of ln(1+e)
//     sp = max(0, t) + residual               // softplus identity
//     out = beta_reciprocal * sp
//
// Real LLK reference:
//   tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softplus.h
#include <cmath>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void softplus_tile_init() {}

ALWI void softplus_tile(uint32_t idst, uint32_t beta, uint32_t beta_reciprocal, uint32_t threshold) {
    __emule_dst_check(idst, "softplus_tile");
    float beta_f;
    float beta_reciprocal_f;
    float threshold_f;
    std::memcpy(&beta_f, &beta, sizeof(float));
    std::memcpy(&beta_reciprocal_f, &beta_reciprocal, sizeof(float));
    std::memcpy(&threshold_f, &threshold, sizeof(float));

    // Residual polynomial coefficients for ln(1+exp(-a)) on [0, 5], degree 8.
    // Verbatim from ckernel_sfpu_softplus.h:30-38.
    constexpr float SOFTPLUS_POLY_C0 =  6.9310557842e-01f;
    constexpr float SOFTPLUS_POLY_C1 = -4.9926245213e-01f;
    constexpr float SOFTPLUS_POLY_C2 =  1.2186349183e-01f;
    constexpr float SOFTPLUS_POLY_C3 =  5.6753782555e-03f;
    constexpr float SOFTPLUS_POLY_C4 = -1.0528374463e-02f;
    constexpr float SOFTPLUS_POLY_C5 =  2.7290175203e-03f;
    constexpr float SOFTPLUS_POLY_C6 = -3.4358495031e-04f;
    constexpr float SOFTPLUS_POLY_C7 =  2.1285692128e-05f;
    constexpr float SOFTPLUS_POLY_C8 = -4.8245715334e-07f;
    constexpr float SOFTPLUS_POLY_BOUNDARY = 5.0f;

    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        float t = beta_f * x;
        if (t > threshold_f) {
            __emule_dst[idst][i] = x;
            continue;
        }

        float a = std::fabs(t);

        // Horner eval of C0 + a*(C1 + a*(C2 + ... + a*C8))
        float residual = SOFTPLUS_POLY_C8;
        residual = residual * a + SOFTPLUS_POLY_C7;
        residual = residual * a + SOFTPLUS_POLY_C6;
        residual = residual * a + SOFTPLUS_POLY_C5;
        residual = residual * a + SOFTPLUS_POLY_C4;
        residual = residual * a + SOFTPLUS_POLY_C3;
        residual = residual * a + SOFTPLUS_POLY_C2;
        residual = residual * a + SOFTPLUS_POLY_C1;
        residual = residual * a + SOFTPLUS_POLY_C0;

        if (a > SOFTPLUS_POLY_BOUNDARY) {
            float e = std::exp(-a);
            residual = e * (1.0f + e * (-0.5f + e * (1.0f / 3.0f)));
        }

        float tmax = (t > 0.0f) ? t : 0.0f;
        float sp = tmax + residual;
        __emule_dst[idst][i] = beta_reciprocal_f * sp;
    }
}

}  // namespace ckernel
