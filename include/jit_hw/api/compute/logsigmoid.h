// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/logsigmoid.h`. Intercepts the upstream include
// path which pulls in `llk_math_eltwise_binary_sfpu_logsigmoid.h` (an LLK-only
// header that references SFPU intrinsics).
//
// logsigmoid(x) = -softplus(-x) = -log(1 + exp(-x)).
// Upstream is a *binary* SFPU op: caller pre-stages exp(-x) into idst_in1,
// then logsigmoid_tile consumes (x, exp(-x)) and writes the result to idst_out.
// The emule path reproduces that contract: read exp(-x) from idst_in1, apply
// -log1p, and store the per-element result in idst_out.
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/logsigmoid.h
//   tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_logsigmoid.h
#include <cmath>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void logsigmoid_tile_init() {}

ALWI void logsigmoid_tile(uint32_t idst_in0, uint32_t idst_in1, uint32_t idst_out) {
    __emule_dst_check(idst_in0, "logsigmoid_tile");
    __emule_dst_check(idst_in1, "logsigmoid_tile");
    __emule_dst_check(idst_out, "logsigmoid_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        // idst_in1 already holds exp(-x) per the upstream binary-SFPU contract.
        float exp_neg_x = __emule_dst[idst_in1][i];
        __emule_dst[idst_out][i] = -std::log1p(exp_neg_x);
    }
}

}  // namespace ckernel
