// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for erfinv SFPU tile op.
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/erfinv.h
// LLK family: llk_math_eltwise_unary_sfpu_erfinv

#include <cmath>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void erfinv_tile_init() {}

ALWI void erfinv_tile(uint32_t idst) {
    __emule_dst_check(idst, "erfinv_tile");
    constexpr float a = 0.147f;
    const float two_over_pi_a = static_cast<float>(2.0 / (M_PI * a));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        float sgn = (x < 0.0f) ? -1.0f : 1.0f;
        float ln = std::log(1.0f - x * x);
        float term = two_over_pi_a + ln / 2.0f;
        __emule_dst[idst][i] = sgn * std::sqrt(std::sqrt(term * term - ln / a) - term);
    }
}

}  // namespace ckernel
