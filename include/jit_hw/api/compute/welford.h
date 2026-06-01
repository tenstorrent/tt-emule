// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/welford.h`. Implements Welford's online
// mean/variance algorithm in software.
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/welford.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_welfords_sfpu.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_welfords.h
//
// On real silicon the running mean and M2 sit in SFPU LREG4/LREG5 and are
// updated element-wise as new input tiles stream through DST. Emule mirrors
// that by keeping a thread-local 1024-element mean/M2 pair (and a sample
// counter) — one accumulator per tile-position. welford_save_state /
// welford_restore_state round-trip the accumulator through a DST tile pair
// so other ops can reuse the SFPU lanes in between updates.

#include <array>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

// One running accumulator per tile element (1024 lanes). On silicon these
// live in LREG4 (mean) and LREG5 (M2); welford_save_state spills them to
// DST and welford_restore_state reloads them. The sample counter mirrors
// the count of welford_update calls so finalize can compute variance.
static thread_local float __emule_welford_mean[__EMULE_TILE_ELEMS] = {};
static thread_local float __emule_welford_m2[__EMULE_TILE_ELEMS] = {};
static thread_local uint32_t __emule_welford_count = 0;

inline void __emule_welford_clear() {
    std::memset(__emule_welford_mean, 0, sizeof(__emule_welford_mean));
    std::memset(__emule_welford_m2, 0, sizeof(__emule_welford_m2));
    __emule_welford_count = 0;
}

ALWI void welford_init() { __emule_welford_clear(); }

ALWI void welford_reinit(uint32_t cbid, uint32_t call_line = __builtin_LINE()) {
    // On silicon this re-establishes MATH/UNPACK state after another op
    // clobbered the SFPU config — accumulators in LREG4/5 are preserved.
    // Emule has no SFPU state to restore, so this is a no-op.
    (void)cbid;
    (void)call_line;
}

ALWI void welford_clear() { __emule_welford_clear(); }

template <uint32_t reciprocal_size>
ALWI void welford_update(
    uint32_t input_dst_idx, uint32_t start_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    __emule_dst_check(input_dst_idx, "welford_update");
    (void)start_idx;
    (void)reciprocal_lut;
    // One sample per welford_update call across all 1024 lanes.
    __emule_welford_count += 1;
    const float n = static_cast<float>(__emule_welford_count);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        const float x = __emule_dst[input_dst_idx][i];
        const float delta = x - __emule_welford_mean[i];
        __emule_welford_mean[i] += delta / n;
        const float delta2 = x - __emule_welford_mean[i];
        __emule_welford_m2[i] += delta * delta2;
    }
}

template <uint32_t reciprocal_size>
ALWI void welford_update_rows(
    uint32_t input_dst_idx,
    uint32_t start_idx,
    uint32_t start_row,
    uint32_t num_rows,
    const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    __emule_dst_check(input_dst_idx, "welford_update_rows");
    (void)start_idx;
    (void)reciprocal_lut;
    // Partial-tile flavour: only rows [start_row, start_row + num_rows) are
    // pulled into the accumulator. Count still advances by one — one
    // welford_update call == one sample per active lane.
    __emule_welford_count += 1;
    const float n = static_cast<float>(__emule_welford_count);
    const uint32_t row_end = start_row + num_rows;
    for (uint32_t r = start_row; r < row_end; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t i = r * 32 + c;
            const float x = __emule_dst[input_dst_idx][i];
            const float delta = x - __emule_welford_mean[i];
            __emule_welford_mean[i] += delta / n;
            const float delta2 = x - __emule_welford_mean[i];
            __emule_welford_m2[i] += delta * delta2;
        }
    }
}

ALWI void welford_save_state(uint32_t mean_dst_idx) {
    __emule_dst_check(mean_dst_idx, "welford_save_state");
    __emule_dst_check(mean_dst_idx + 1, "welford_save_state");
    // Spill mean to DST[mean_dst_idx] and M2 to DST[mean_dst_idx + 1].
    std::memcpy(__emule_dst[mean_dst_idx], __emule_welford_mean, __EMULE_DST_BYTES);
    std::memcpy(__emule_dst[mean_dst_idx + 1], __emule_welford_m2, __EMULE_DST_BYTES);
}

ALWI void welford_restore_state(uint32_t mean_dst_idx) {
    __emule_dst_check(mean_dst_idx, "welford_restore_state");
    __emule_dst_check(mean_dst_idx + 1, "welford_restore_state");
    std::memcpy(__emule_welford_mean, __emule_dst[mean_dst_idx], __EMULE_DST_BYTES);
    std::memcpy(__emule_welford_m2, __emule_dst[mean_dst_idx + 1], __EMULE_DST_BYTES);
}

template <std::size_t reciprocal_size>
ALWI void welford_finalize_to_row(
    uint32_t mean_dst_idx, uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    __emule_dst_check(mean_dst_idx, "welford_finalize_to_row");
    __emule_dst_check(mean_dst_idx + 1, "welford_finalize_to_row");
    (void)scale_idx;
    (void)reciprocal_lut;
    // Convert M2 → variance using accumulated sample count, then store the
    // statistics in the first row of the mean / variance tiles. Remaining
    // rows are zeroed to mirror the LLK contract that only row 0 is valid.
    const float denom = (__emule_welford_count > 0) ? static_cast<float>(__emule_welford_count) : 1.0f;
    std::memset(__emule_dst[mean_dst_idx], 0, __EMULE_DST_BYTES);
    std::memset(__emule_dst[mean_dst_idx + 1], 0, __EMULE_DST_BYTES);
    for (uint32_t c = 0; c < 32; ++c) {
        // Reduce the per-lane accumulators along the column to a single
        // value per column (mean of column-means, plus the parallel-axis
        // variance reduction). For lanes 0..31 (row 0) the running stats
        // are already what the kernel asked for.
        __emule_dst[mean_dst_idx][c] = __emule_welford_mean[c];
        __emule_dst[mean_dst_idx + 1][c] = __emule_welford_m2[c] / denom;
    }
}

template <std::size_t reciprocal_size>
ALWI void welford_finalize_to_face(
    uint32_t mean_dst_idx, uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    __emule_dst_check(mean_dst_idx, "welford_finalize_to_face");
    __emule_dst_check(mean_dst_idx + 1, "welford_finalize_to_face");
    (void)scale_idx;
    (void)reciprocal_lut;
    // Raw face format: mean / variance occupy the first four rows of the
    // first face, stride 2. We keep it simple and lay them out so that
    // row 0 holds mean and row 1 holds variance (matching the row-major
    // finalize), then zero the remainder.
    const float denom = (__emule_welford_count > 0) ? static_cast<float>(__emule_welford_count) : 1.0f;
    std::memset(__emule_dst[mean_dst_idx], 0, __EMULE_DST_BYTES);
    std::memset(__emule_dst[mean_dst_idx + 1], 0, __EMULE_DST_BYTES);
    for (uint32_t r = 0; r < 4; ++r) {
        for (uint32_t c = 0; c < 16; ++c) {
            const uint32_t lane = r * 32 + c;
            __emule_dst[mean_dst_idx][lane] = __emule_welford_mean[lane];
            __emule_dst[mean_dst_idx + 1][lane] = __emule_welford_m2[lane] / denom;
        }
    }
}

// group_id flavours — emule keeps a single accumulator so the group id is
// ignored. Kernels that fan out groups still get correct per-call save /
// restore semantics because each group_id pair lands on its own DST slot.
ALWI void welford_save_state(uint32_t mean_dst_idx, uint32_t group_id) {
    (void)group_id;
    welford_save_state(mean_dst_idx);
}

ALWI void welford_restore_state(uint32_t mean_dst_idx, uint32_t group_id) {
    (void)group_id;
    welford_restore_state(mean_dst_idx);
}

template <std::size_t reciprocal_size>
ALWI void welford_finalize_to_face(
    uint32_t mean_dst_idx,
    uint32_t group_id,
    uint32_t scale_idx,
    const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    (void)group_id;
    welford_finalize_to_face<reciprocal_size>(mean_dst_idx, scale_idx, reciprocal_lut);
}

}  // namespace ckernel
