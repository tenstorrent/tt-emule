// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// sdpa_sub_custom (column-broadcast subtract) compute shim for emulated mode.
//
// The real api/compute/experimental/sdpa_sub_custom.h pulls in the experimental LLK
// custom eltwise-binary headers (raw-TTI) which emule does not compile. The op is a
// column-broadcast subtract used by SDPA streaming to subtract the per-row max
// (a column vector) from a block of ct_dim score tiles — numerically identical to the
// standard sub_tiles_bcast<ELWSUB, COL>. The "custom" path is a HW execution variant
// with no numerical effect, so delegate to emule's bcast model.
//
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/experimental/llk_math_eltwise_binary_custom.h

#include "jit_hw/api/compute/bcast.h"

namespace ckernel {

// HW unpack/math state config — no-op in emule.
ALWI void sub_bcast_cols_init_short_custom(
    uint32_t /*icb0*/, uint32_t /*icb1*/, uint32_t /*ct_dim*/, uint32_t /*call_line*/ = __builtin_LINE()) {}

// in1 (column vector at itile1) is broadcast across columns and subtracted from
// ct_dim consecutive in0 tiles (itile0 + t) into DST[idst + t]. itile1 is fixed:
// the same per-row max is reused for every column tile of the row (see the
// compute_streaming caller).
ALWI void sub_tiles_bcast_cols_custom(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t ct_dim) {
    for (uint32_t t = 0; t < ct_dim; t++) {
        sub_tiles_bcast<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(icb0, icb1, itile0 + t, itile1, idst + t);
    }
}

}  // namespace ckernel
