// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/ema.h` — exponential moving average SFPU op.
// Silicon contract (tt_metal/hw/inc/api/compute/ema.h):
//   ema_init(alpha, beta);              // alpha/beta are fp32 bit patterns
//   ema_clear_previous_output();        // reset the carried previous output
//   ema_tile(input_dst_index);          // EMA on DST[idst] -> DST[idst+1]
// EMA recurrence (ckernel_sfpu_ema.h): EMA_new = alpha*EMA_old + beta*input.
//
// The kernel wraps ema_tile in its own transpose_wh_tile pair (api/compute/
// transpose_wh.h) so that, in DST, the time axis T runs DOWN the rows and the
// channels run across the columns (issue #53: "transpose_wh maps T->rows").
// ema_tile must therefore chain the recurrence DOWN the rows (successive T) with
// a per-column (per-channel) previous-output carried across rows within the tile
// AND across tiles, reset by ema_clear_previous_output at each batch boundary.
//
// The silicon `_calculate_ema_tile_` brackets the MAD chain with two SFPTRANSP
// ops; those only re-orient how the 32-lane LREGs are loaded/stored and net to
// identity on the DST tile layout, so we do NOT transpose here — operating
// directly on the row-major DST is the correct net behavior. (An earlier draft
// that transposed in/out chained across channels instead of T and dropped the
// alpha*prev carry: out came out as beta*input only.)

#include "jit_hw/api/compute/common.h"

#include <cstdint>
#include <cstring>

namespace ckernel {

// alpha (weight on previous output) and beta (weight on current input), plus the
// per-column carried previous output. Persist across ema_tile calls; reset by
// ema_clear_previous_output at each batch boundary.

ALWI void ema_init(uint32_t alpha, uint32_t beta) {
    std::memcpy(&__emule_compute_ctx().ema_alpha, &alpha, sizeof(float));
    std::memcpy(&__emule_compute_ctx().ema_beta, &beta, sizeof(float));
}

ALWI void ema_clear_previous_output() {
    for (uint32_t j = 0; j < 32; ++j) {
        __emule_compute_ctx().ema_prev[j] = 0.0f;
    }
}

// EMA_new = beta*input + alpha*EMA_old, chaining down rows (T) per column.
ALWI void ema_tile(uint32_t input_dst_index) {
    __emule_dst_check(input_dst_index, "ema_tile.in");
    __emule_dst_check(input_dst_index + 1, "ema_tile.out");
    const float alpha = __emule_compute_ctx().ema_alpha;
    const float beta = __emule_compute_ctx().ema_beta;

    const float* in = __emule_compute_ctx().dst[input_dst_index];
    __emule_dst_mark_dirty(input_dst_index + 1);
    float* out = __emule_compute_ctx().dst[input_dst_index + 1];
    for (uint32_t r = 0; r < 32; ++r) {       // rows = successive time samples T
        for (uint32_t c = 0; c < 32; ++c) {   // cols = channels (independent EMAs)
            const float v = beta * in[r * 32 + c] + alpha * __emule_compute_ctx().ema_prev[c];
            out[r * 32 + c] = v;
            __emule_compute_ctx().ema_prev[c] = v;
        }
    }
}

}  // namespace ckernel
