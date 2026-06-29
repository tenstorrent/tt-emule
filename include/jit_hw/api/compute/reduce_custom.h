// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// reduce_custom (reduce_block_max_row family) compute shim for emulated mode.
//
// The real api/compute/reduce_custom.h pulls in the experimental LLK reduce-custom
// headers (raw-TTI Tensix reduce engine) which emule does not compile. Upstream's own
// docstring states reduce_block_max_row is numerically equivalent to looping the
// standard reduce<MAX, REDUCE_ROW> over the block — the "custom" part is purely a HW
// MOP-split / semaphore-trigger utilization optimization (respect_trigger) with no
// numerical effect. So delegate to emule's reduce_tile<MAX, REDUCE_ROW>; the trigger /
// fp32-accumulation flags and the init/uninit pipeline config are HW-only no-ops.
//
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/experimental/llk_math_reduce_custom.h

#include "jit_hw/api/compute/reduce.h"

namespace ckernel {

template <uint32_t block_ct_dim, bool respect_trigger = false>
ALWI void reduce_block_max_row_init(uint32_t ocb) {}

// Per-row MAX across block_ct_dim consecutive tiles of icb starting at row_start_index,
// scaled by icb_scaler, into DST[idst] (REDUCE_ROW layout: col 0 of each row). reduce_tile
// overwrites a fresh DST slot on the first tile and MAX-accumulates after — so this also
// composes with a preceding copy_tile of a running max (which dirties the slot).
template <uint32_t block_ct_dim, bool respect_trigger = false>
ALWI void reduce_block_max_row(uint32_t icb, uint32_t icb_scaler, uint32_t row_start_index, uint32_t idst) {
    for (uint32_t t = 0; t < block_ct_dim; t++) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(icb, icb_scaler, row_start_index + t, 0, idst);
    }
}

template <bool clear_fp32_accumulation = false, bool respect_trigger = false>
ALWI void reduce_block_max_row_uninit(uint32_t icb) {}

// Runtime variants — block_ct_dim is a runtime arg configured by the _init_runtime
// call and consumed by the execute call (on silicon it lives in the unpack/MOP HW
// state programmed by init; the execute replays it). emule carries it across the
// pair in a thread_local, mirroring that config-then-execute handoff.
static thread_local uint32_t __emule_reduce_block_ct_dim = 1;

ALWI void reduce_block_max_row_init_runtime(uint32_t ocb, uint32_t block_ct_dim, bool respect_trigger = false) {
    __emule_reduce_block_ct_dim = block_ct_dim;
}

// respect_trigger + overlap_first_half are silicon MOP-split / unpacker-scheduling
// optimizations; they have no effect on the row-major reduce emule computes. The
// signature mirrors the upstream tt_metal/hw/inc/api/compute/reduce_custom.h so SDPA's
// compute_streaming.hpp (which passes overlap_first_half) resolves against the shadow.
ALWI void reduce_block_max_row_runtime(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t row_start_index,
    uint32_t idst,
    bool respect_trigger = false,
    bool overlap_first_half = false) {
    for (uint32_t t = 0; t < __emule_reduce_block_ct_dim; t++) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(icb, icb_scaler, row_start_index + t, 0, idst);
    }
}

ALWI void reduce_block_max_row_uninit_runtime(
    uint32_t icb, bool respect_trigger = false, bool overlap_first_half = false) {}

}  // namespace ckernel

using namespace ckernel;
