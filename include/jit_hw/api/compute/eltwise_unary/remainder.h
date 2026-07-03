// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/remainder.h` (scalar `remainder_tile`).
// `param0` = uint32 bit-pattern of the fp32 divisor; `param1` = the host-computed
// fp32 reciprocal `fl(1/divisor)` (load-bearing — see below).
//
// Real LLK: ckernel_sfpu_remainder.h `calculate_remainder`. The scalar path has no
// runtime Newton: silicon uses the host reciprocal + an exp-derived mantissa
// mask-floor of `|v| * recip`, NOT an exact `floor(|v|/divisor)`. At multiples of a
// tiny divisor the rounded reciprocal makes the mask-floor land one below the exact
// quotient, so the remainder is one divisor off — which is what the bf16 golden was
// calibrated to. Mirror the SFPU sequence in scalar fp32 (no double promotion).
#include <cmath>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void remainder_tile_init(uint32_t param0, uint32_t param1) {
    // No SFPU state to initialize: emule reads param0/param1 directly in
    // remainder_tile rather than staging them into vConstFloatPrgm0/1.
    (void)param0;
    (void)param1;
}

ALWI void remainder_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    __emule_dst_check(idst, "remainder_tile");
    float divisor, recip;
    std::memcpy(&divisor, &param0, sizeof(float));
    std::memcpy(&recip, &param1, sizeof(float));
    const float s = std::fabs(divisor);
    const float r = std::fabs(recip);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        const float val = __emule_compute_ctx().dst[idst][i];
        if (s == 0.0f) {
            // Divisor 0: silicon yields a NaN that packs to bf16 -inf; the test maps
            // that inf back to nan. Same bit pattern as binary_remainder.h.
            const uint32_t nan_bits = 0xFF800001u;
            std::memcpy(&__emule_compute_ctx().dst[idst][i], &nan_bits, sizeof(float));
            continue;
        }
        float v = std::fabs(val);
        const float prod = v * r;  // fp32, matches silicon's mul (no FMA)
        float quotient;
        if (prod == 0.0f) {
            quotient = 0.0f;
        } else {
            int e;
            std::frexp(prod, &e);
            e -= 1;  // unbiased exponent of prod (exexp)
            if (e < 0) {
                quotient = 0.0f;
            } else if (e < 23) {
                // Clear the low (23-e) mantissa bits → floor toward zero.
                uint32_t u;
                std::memcpy(&u, &prod, sizeof(u));
                u &= ~((1u << (23 - e)) - 1u);
                std::memcpy(&quotient, &u, sizeof(quotient));
            } else {
                quotient = prod;
            }
            if (quotient > prod) {
                quotient -= 1.0f;  // mask-edge guard
            }
        }
        const float qs = quotient * s;
        v = v - qs;
        if (val < 0.0f && v != 0.0f) {
            v = s - v;
        }
        if (divisor < 0.0f && v != 0.0f) {
            v = v + divisor;
        }
        v = std::copysign(v, divisor);
        for (int l = 0; l < 10; l++) {
            if (v >= s) {
                v = s - v;
            }
        }
        if (std::fabs(v) - s == 0.0f) {
            v = 0.0f;
        }
        __emule_compute_ctx().dst[idst][i] = v;
    }
}

}  // namespace ckernel
