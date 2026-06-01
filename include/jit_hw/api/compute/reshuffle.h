// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/reshuffle.h`. Intercepts the upstream include
// path which pulls in `llk_math_eltwise_unary_sfpu_reshuffle_rows.h` (an
// LLK-only header that references SFPU intrinsics).
//
// Semantics: for each row r in [0, 32), copy input row indices[r] into output
// row r. The `addr` argument is a uint32 L1 address holding 32 uint32 row
// indices. We treat the address as a raw host pointer (matches dataflow_api's
// L1 access pattern) and use a temp buffer + memcpy to avoid in-place
// aliasing when an index maps to a different row.
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/reshuffle.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_reshuffle_rows.h

#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void reshuffle_rows_tile_init() {}

ALWI void reshuffle_rows_tile(uint32_t idst, uint32_t addr) {
    const uint32_t* indices = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(addr));

    // Snapshot source rows into a temp buffer so an out-of-order permutation
    // (e.g. indices[r] != r for multiple r) cannot clobber rows we still
    // need to read.
    float temp[1024];
    std::memcpy(temp, __emule_dst[idst], sizeof(temp));

    for (uint32_t r = 0; r < 32; r++) {
        uint32_t src_row = indices[r];
        std::memcpy(&__emule_dst[idst][r * 32], &temp[src_row * 32], 32 * sizeof(float));
    }
}

}  // namespace ckernel
