// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for api/compute/reduce_custom.h. The real header chains into
// llk_math_reduce_api.h + experimental LLK headers that reference RISC-V
// SFPU vector intrinsics not available on x86. Provide no-op + minimal
// host-side reduce_block_max_row stubs.
//
// SDPA flash-decode is the main consumer. The reduce_block_max_row operation
// finds the per-row max across a block of tiles — required for numerically
// stable softmax (subtract-max). For functional correctness in emule, the
// implementation computes the actual max via direct CB reads.

#include <cstdint>
#include <cstring>
#include <limits>
#include "api/compute/common.h"

namespace ckernel {

ALWI void reduce_block_max_row_init() {}
ALWI void reduce_block_max_row_reinit_short() {}
ALWI void reduce_block_max_row_reinit_minimal() {}
ALWI void reduce_block_max_row_reinit_minimal_runtime(uint32_t /*block_ct_dim*/, bool /*respect_trigger*/ = false) {}
ALWI void reduce_block_max_row_reinit_short_runtime(uint32_t /*block_ct_dim*/, bool /*respect_trigger*/ = false) {}
ALWI void reduce_block_max_row_uninit(uint32_t /*icb*/) {}
ALWI void reduce_block_max_row_init_runtime(uint32_t /*block_ct_dim*/, bool /*respect_trigger*/ = false) {}
ALWI void reduce_block_max_row_uninit_runtime(uint32_t /*icb*/, bool /*respect_trigger*/ = false) {}

// reduce_block_max_row: For each row r of the 32-row tile, compute
//   dst[idst][r, 0] = max(icb[t, r, c]) over all tiles t in [row_start, row_start+block_ct_dim)
//                                                and all columns c in [0, 32).
// Real semantics: scaler is 1.0; output goes to column 0 of dst tile.
ALWI void reduce_block_max_row(uint32_t icb, uint32_t /*icb_scaler*/,
                               uint32_t row_start_index, uint32_t idst) {
    __emule_dst_check(idst, "reduce_block_max_row");
    constexpr uint32_t DIM = 32;
    for (uint32_t r = 0; r < DIM; r++) {
        float m = -std::numeric_limits<float>::infinity();
        // Number of tiles to reduce over is encoded in the previous init call;
        // since we can't get block_ct_dim here from a previous init, treat as
        // single-tile reduction (most decode-path SDPA calls invoke this once
        // per block iter).
        for (uint32_t c = 0; c < DIM; c++) {
            float v = __emule_read_cb_elem_at(icb, row_start_index, r * DIM + c);
            if (v > m) m = v;
        }
        __emule_dst[idst][r * DIM] = m;
    }
}

ALWI void reduce_block_max_row_runtime(uint32_t icb, uint32_t icb_scaler,
                                        uint32_t row_start_index, uint32_t idst,
                                        uint32_t /*block_ct_dim*/ = 1,
                                        bool /*respect_trigger*/ = false) {
    reduce_block_max_row(icb, icb_scaler, row_start_index, idst);
}

}  // namespace ckernel
