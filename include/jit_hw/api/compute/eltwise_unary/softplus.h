// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/softplus.h`. Intercepts the
// upstream include path which pulls in `ckernel_sfpu_softplus.h` (an LLK-only
// header that references SFPU intrinsics).
//
// Semantics (per element):
//   beta, beta_reciprocal, threshold are fp32 bit-patterns packed into uint32.
//   out = (beta*x > threshold) ? x : log1p(exp(beta*x)) * beta_reciprocal.
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/eltwise_unary/softplus.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_softplus.h
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
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        float bx = beta_f * x;
        if (bx > threshold_f) {
            __emule_dst[idst][i] = x;
        } else {
            __emule_dst[idst][i] = std::log1p(std::exp(bx)) * beta_reciprocal_f;
        }
    }
}

}  // namespace ckernel
