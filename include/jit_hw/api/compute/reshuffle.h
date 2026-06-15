// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/reshuffle.h`. Intercepts the upstream include
// path which pulls in `llk_math_eltwise_unary_sfpu_reshuffle_rows.h` (an
// LLK-only header that references SFPU intrinsics).
//
// Semantics (per silicon _calculate_reshuffle_rows_): `addr` is the L1
// address of a mask tile MINUS the 16-byte tile header (callers pass
// idx_addr - 16; the LLK reads indices at addr + 16). The mask is 32 uint8
// destination-row indices: for each input row r, output_tile[idx[r]] +=
// input_tile[r], skipping rows with idx[r] >= 32 (255 = invalid sentinel).
// Input rows come from DST tile `idst`, accumulation goes into DST tile
// `idst + 1` (silicon's output_tile_offset of 64 dest rows = next tile);
// the caller pre-loads tile idst + 1 and packs from it.
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/reshuffle.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_reshuffle_rows.h

#include <cstdint>

#include "jit_hw/api/compute/common.h"
#include "jit_hw/jit_kernel_stubs.hpp"  // __emule_local_l1_to_ptr

namespace ckernel {

ALWI void reshuffle_rows_tile_init() {}

ALWI void reshuffle_rows_tile(uint32_t idst, uint32_t addr) {
    // +16 skips the tile header, mirroring the LLK's `idx_addr + 16`.
    const uint8_t* idx = __emule_local_l1_to_ptr(addr) + 16;

    for (uint32_t r = 0; r < 32; r++) {
        uint32_t dst_row = idx[r];
        if (dst_row >= 32) {
            continue;
        }
        for (uint32_t c = 0; c < 32; c++) {
            __emule_dst[idst + 1][dst_row * 32 + c] += __emule_dst[idst][r * 32 + c];
        }
    }
}

}  // namespace ckernel
