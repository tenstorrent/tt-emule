// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for lgamma SFPU tile ops. Ports the upstream four-region
// polynomial approximation (Taylor around z=1, Taylor around z=2,
// Stirling + Bernoulli everywhere else). Reflection correction for x<0.5
// is applied separately by lgamma_adjusted_tile via input idst1
// (= log|sin(pi*frac(x))|).
//
// Real LLK reference:
//   tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h
// Constants verbatim from that file.

#include <cmath>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

namespace {

// lgamma(z) for z >= ~0.5 using the upstream three-region polynomial.
inline float lgamma_eval_z(float z) {
    constexpr float LOG_SQRT_2PI = 0.9189385332046727f;
    constexpr float LOG_SQRT_PI  = 0.57236494f;
    constexpr float r0 =  0.0833333333f;   //  1/12
    constexpr float r1 = -0.0027777777f;   // -1/360
    constexpr float p0 = -0.57721566f, p1 =  0.82246703f, p2 = -0.40068563f;
    constexpr float p3 =  0.27058081f, p4 = -0.20738555f;
    constexpr float q0 =  0.42278434f, q1 =  0.32246703f, q2 = -0.06735230f;
    constexpr float q3 =  0.02058081f, q4 = -0.00738555f;

    if (std::fabs(z - 0.5f) < 0.01f) return LOG_SQRT_PI;
    if (z == 1.0f || z == 2.0f)      return 0.0f;

    float d1 = z - 1.0f;
    if (std::fabs(d1) < 0.25f) {
        float r = p4;
        r = r * d1 + p3;
        r = r * d1 + p2;
        r = r * d1 + p1;
        r = r * d1 + p0;
        return d1 * r;
    }
    float d2 = z - 2.0f;
    if (std::fabs(d2) < 0.25f) {
        float r = q4;
        r = r * d2 + q3;
        r = r * d2 + q2;
        r = r * d2 + q1;
        r = r * d2 + q0;
        return d2 * r;
    }
    // Stirling + Bernoulli correction.
    float inv_z = 1.0f / z;
    float res = (z - 0.5f) * std::log(z) - z + LOG_SQRT_2PI
              + inv_z * (r0 + (inv_z * inv_z) * r1);
    return res;
}

}  // anon

ALWI void lgamma_stirling_tile_init() {}

// Computes lgamma(z) where z = (x < 0.5) ? 1 - x : x. Reflection correction
// for x < 0.5 is applied by lgamma_adjusted_tile.
ALWI void lgamma_stirling_tile(uint32_t idst) {
    __emule_dst_check(idst, "lgamma_stirling_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        float z = (x < 0.5f) ? (1.0f - x) : x;
        __emule_compute_ctx().dst[idst][i] = lgamma_eval_z(z);
    }
}

ALWI void lgamma_stirling_float_tile_init() {}

// Float32-input variant. idst0 holds x; idst1 (precomputed log z on silicon)
// is ignored in emule — we recompute via std::log. Writes lgamma(z) into idst2.
ALWI void lgamma_stirling_float_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2) {
    __emule_dst_check(idst0, "lgamma_stirling_float_tile");
    __emule_dst_check(idst1, "lgamma_stirling_float_tile");
    __emule_dst_check(idst2, "lgamma_stirling_float_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst0][i];
        float z = (x < 0.5f) ? (1.0f - x) : x;
        __emule_compute_ctx().dst[idst2][i] = lgamma_eval_z(z);
    }
}

ALWI void lgamma_adjusted_tile_init() {}

// Combines lgamma_stirling result with the reflection correction.
//   idst0: lgamma_stirling result (lgamma(z))
//   idst1: log|sin(pi * frac(x))| (reflection correction term)
//   idst2: original input x
//   idst3: output lgamma(x)
ALWI void lgamma_adjusted_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t idst3) {
    __emule_dst_check(idst0, "lgamma_adjusted_tile");
    __emule_dst_check(idst1, "lgamma_adjusted_tile");
    __emule_dst_check(idst2, "lgamma_adjusted_tile");
    __emule_dst_check(idst3, "lgamma_adjusted_tile");
    constexpr float ln_pi = 1.1447298858f;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x            = __emule_compute_ctx().dst[idst2][i];
        float res_stirling = __emule_compute_ctx().dst[idst0][i];
        float log_sin_pi_x = __emule_compute_ctx().dst[idst1][i];
        float reflection_adj = ln_pi - log_sin_pi_x;
        __emule_compute_ctx().dst[idst3][i] = (x < 0.5f) ? (reflection_adj - res_stirling) : res_stirling;
    }
}

}  // namespace ckernel

// The lgamma kernels declare their own `constexpr float M_PI`, which collides
// with the M_PI macro from <cmath>. Undef it here (this header is included
// before their definition); no emule code uses the macro.
#ifdef M_PI
#undef M_PI
#endif
