// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for api/compute/eltwise_unary/tanh_derivative.h (tanh backward).
// sech^2(x) = 4*exp(-2|x|) / (1 + exp(-2|x|))^2 - cancellation-safe vs 1 - tanh^2(x).
// Real LLK: ckernel_sfpu_tanh_derivative.h::calculate_tanh_derivative_sech2.
#include "jit_hw/api/compute/common.h"

#include <cmath>
#include <cstdint>

namespace ckernel {

template <bool fast_and_approx = false>
ALWI void tanh_derivative_tile_init() {}

template <bool fast_and_approx = false>
ALWI void tanh_derivative_tile(uint32_t idst) {
    __emule_dst_check(idst, "tanh_derivative_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        const float e = std::exp(-2.0f * std::fabs(__emule_dst[idst][i]));
        const float d = 1.0f + e;
        __emule_dst[idst][i] = 4.0f * e / (d * d);
    }
}

}  // namespace ckernel
