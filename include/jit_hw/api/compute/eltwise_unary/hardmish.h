// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for SFPU hardmish tile op.
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/hardmish.h
//
// hardmish(x) = x * clamp(x + 2, 0, 2) / 2
//             = x * clamp(0.5 * x + 1, 0, 1)
// Piecewise (finite x):
//   x <= -2  =>  0
//   x >= 0   =>  x
//   else     =>  x * (x + 2) / 2

#include <algorithm>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void hardmish_tile_init() {}

ALWI void hardmish_tile(uint32_t idst) {
    __emule_dst_check(idst, "hardmish_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        float clamped = std::fmin(std::fmax(x + 2.0f, 0.0f), 2.0f);
        __emule_compute_ctx().dst[idst][i] = x * clamped * 0.5f;
    }
}

} // namespace ckernel
