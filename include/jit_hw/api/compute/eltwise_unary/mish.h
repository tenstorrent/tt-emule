// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/mish.h`. Intercepts the upstream
// include path which pulls in `llk_math_eltwise_unary_sfpu_mish.h` (an LLK-only
// header that references SFPU intrinsics).
//
// Semantics (per element):
//   mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/eltwise_unary/mish.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_mish.h
#include <cstdint>
#include <cmath>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void mish_tile_init() {}

ALWI void mish_tile(uint32_t idst) {
    __emule_dst_check(idst, "mish_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        // softplus(x) = log(1 + exp(x)); use log1pf+expf for numerical stability
        // on small/negative x. For very large positive x, mish(x) -> x.
        float sp = std::log1p(std::exp(x));
        __emule_dst[idst][i] = x * std::tanh(sp);
    }
}

}  // namespace ckernel
