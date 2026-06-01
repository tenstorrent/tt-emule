// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for i1 SFPU tile op.
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/i1.h
// LLK family: llk_math_eltwise_unary_sfpu_i1

#include <cmath>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void i1_tile_init() {}

ALWI void i1_tile(uint32_t idst) {
    __emule_dst_check(idst, "i1_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_dst[idst][i] = std::cyl_bessel_i(1, __emule_dst[idst][i]);
    }
}

}  // namespace ckernel
