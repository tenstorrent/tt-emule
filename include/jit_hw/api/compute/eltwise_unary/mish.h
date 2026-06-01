// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/mish.h`. Intercepts the upstream
// include path which pulls in `llk_math_eltwise_unary_sfpu_mish.h` (an LLK-only
// header that references SFPU intrinsics).
//
// Semantics (per element), matching upstream's fused algebraic identity:
//   u       = exp(x)
//   numer   = u * (u + 2)
//   denom   = u^2 + 2u + 2   (i.e. (1 + u)^2 + 1)
//   mish(x) = x * numer / denom
// Saturation: for x >= 8.0f, mish(x) ≈ x.
//
// Real LLK reference:
//   tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mish.h
#include <cstdint>
#include <cmath>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void mish_tile_init() {}

ALWI void mish_tile(uint32_t idst) {
    __emule_dst_check(idst, "mish_tile");
    constexpr float SAT_HI = 8.0f;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        float result;
        if (x >= SAT_HI) {
            result = x;
        } else {
            float u = std::exp(x);
            float numer = u * (u + 2.0f);
            float denom = u * u + 2.0f * u + 2.0f;
            result = x * numer / denom;
        }
        __emule_dst[idst][i] = result;
    }
}

}  // namespace ckernel
