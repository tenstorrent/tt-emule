// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for cbrt SFPU tile op.
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h
// LLK family: llk_math_eltwise_unary_sfpu_cbrt

#include <cmath>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void cbrt_tile_init() {}

ALWI void cbrt_tile(uint32_t idst) {
    __emule_dst_check(idst, "cbrt_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_compute_ctx().dst[idst][i] = std::cbrt(__emule_compute_ctx().dst[idst][i]);
    }
}

}  // namespace ckernel
