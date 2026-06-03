// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/xielu.h`. xIELU (Expanded
// Integral of ELU; arXiv:2411.13010), beta hard-coded to 0.5:
//   x > 0 : alpha_p * x^2 + beta * x
//   x <= 0: alpha_n * (expm1(min(x, -1e-6)) - x) + beta * x
// alpha_p / alpha_n are uint32 bit-patterns of fp32.
// Real LLK: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_xielu.h

#include <cmath>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void xielu_tile_init() {}

ALWI void xielu_tile(uint32_t idst, uint32_t alpha_p, uint32_t alpha_n) {
    __emule_dst_check(idst, "xielu_tile");

    float ap;
    float an;
    std::memcpy(&ap, &alpha_p, sizeof(float));
    std::memcpy(&an, &alpha_n, sizeof(float));

    constexpr float kBeta = 0.5f;
    constexpr float kEps = -1e-6f;  // LLK's vConstFloatPrgm1

    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        float beta_mul_x = kBeta * x;
        float y;
        if (x > 0.0f) {
            // Positive branch: alpha_p * x^2 + beta * x
            y = ap * x * x + beta_mul_x;
        } else {
            // Negative branch: alpha_n * (expm1(min(x, eps)) - x) + beta * x.
            // Clamping the expm1 argument to <= eps matches the LLK piecewise
            // dispatch (very-small-negative region uses expm1(eps) directly).
            float arg = x < kEps ? x : kEps;
            float exp_term = std::expm1(arg) - x;
            y = an * exp_term + beta_mul_x;
        }
        __emule_dst[idst][i] = y;
    }
}

}  // namespace ckernel
