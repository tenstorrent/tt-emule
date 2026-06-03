// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for digamma SFPU tile op.
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/digamma.h
// LLK: llk_math_eltwise_unary_sfpu_digamma (ckernel_sfpu_digamma::calculate_digamma)

#include <cmath>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void digamma_tile_init() {}

// Computes psi(x) = d/dx log(Gamma(x)) using the recurrence
//   psi(x) = psi(x + 1) - 1/x
// to shift x into the asymptotic regime (x > 6), then applies the
// Stirling asymptotic series
//   psi(x) ~ log(x) - 1/(2x) - 1/(12 x^2) + 1/(120 x^4) - 1/(252 x^6).
ALWI void digamma_tile(uint32_t idst) {
    __emule_dst_check(idst, "digamma_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        float result = 0.0f;
        // Recurrence: shift x up until x > 6 so the asymptotic series is accurate.
        while (x <= 6.0f) {
            result -= 1.0f / x;
            x += 1.0f;
        }
        float inv = 1.0f / x;
        float inv2 = inv * inv;
        float inv4 = inv2 * inv2;
        float inv6 = inv4 * inv2;
        result += std::log(x) - 0.5f * inv
                  - (1.0f / 12.0f) * inv2
                  + (1.0f / 120.0f) * inv4
                  - (1.0f / 252.0f) * inv6;
        __emule_dst[idst][i] = result;
    }
}

}  // namespace ckernel
