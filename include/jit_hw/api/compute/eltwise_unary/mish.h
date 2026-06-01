// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/mish.h`. Upstream's fused form:
//   u = exp(x); mish(x) = x * u * (u + 2) / (u^2 + 2u + 2). x>=8.0 saturates.
// Real LLK: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mish.h
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
