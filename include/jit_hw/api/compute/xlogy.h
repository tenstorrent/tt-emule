// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void xlogy_binary_tile_init() {}

ALWI void xlogy_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst0][i];
        float y = __emule_compute_ctx().dst[idst1][i];
        __emule_compute_ctx().dst[odst][i] = (x == 0.f) ? 0.f : x * std::log(y);
    }
}

} // namespace ckernel
