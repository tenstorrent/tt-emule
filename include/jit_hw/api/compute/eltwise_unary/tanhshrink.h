// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/tanhshrink.h`. Upstream form:
//   tanhshrink(x) = x - tanh(x).
// Real LLK: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanhshrink.h
// (silicon uses bf16/fp32 minimax polynomials to dodge catastrophic cancellation
// for small |x|; emule's DST is fp32, so the direct closed form is faithful.)
#include <cstdint>
#include <cmath>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

template <bool APPROXIMATION_MODE = false>
ALWI void tanhshrink_tile_init() {}

template <bool APPROXIMATION_MODE = false>
ALWI void tanhshrink_tile(uint32_t idst) {
    __emule_dst_check(idst, "tanhshrink_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        __emule_compute_ctx().dst[idst][i] = x - std::tanh(x);
    }
}

}  // namespace ckernel
