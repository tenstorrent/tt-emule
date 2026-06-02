// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/cumsum.h`. Columnwise prefix sum within a tile.
// Multi-tile cumsum arrives in NWH order; `first=true` resets the per-column
// accumulator at H=0, `first=false` continues from the previous tile.
// Real LLK: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

// 32 per-column running accumulators. Persists across cumsum_tile calls; the
// `first` argument resets it at the start of each NWH column-of-tiles.
static thread_local float __emule_cumsum_acc[32] = {0};

ALWI void cumsum_tile_init() {
    for (uint32_t c = 0; c < 32; c++) {
        __emule_cumsum_acc[c] = 0.0f;
    }
}

ALWI void cumsum_tile(uint32_t idst, bool first = true) {
    __emule_dst_check(idst, "cumsum_tile");
    if (first) {
        for (uint32_t c = 0; c < 32; c++) {
            __emule_cumsum_acc[c] = 0.0f;
        }
    }
    // Columnwise prefix sum: for each column c, walk rows 0..31 top-to-bottom,
    // adding the running per-column accumulator.
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            __emule_cumsum_acc[c] += __emule_dst[idst][r * 32 + c];
            __emule_dst[idst][r * 32 + c] = __emule_cumsum_acc[c];
        }
    }
}

}  // namespace ckernel
