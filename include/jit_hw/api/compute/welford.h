// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/welford.h`. Welford's online mean/variance.
//
// Silicon model (tt_metal/tt-llk/.../ckernel_sfpu_welfords.h): welford_update
// reduces DOWN the rows of the input DST tile — each of the 32 rows is one
// sample of 32 column-values — into a PER-COLUMN running mean/M2 accumulator
// (LREG4/LREG5, 32 lanes). The k-th processed row is sample number
// (start_idx + k + 1); the reciprocal LUT supplies 1/(start_idx+k+1). The caller
// tracks the count via start_idx (per update) and scale_idx (at finalize), so
// emule keeps NO internal count — it derives 1/n directly from those indices
// (equivalent to the standard LUT [1/1, 1/2, ...], and matching the LLK fallback
// 1/(idx+1) when idx >= lut size).
//
// Consumers: layernorm-welford uses one accumulator across the W-tile loop
// (welford_init -> welford_update* -> welford_finalize_to_row). groupnorm-welford
// keeps one accumulator PER GROUP, swapping them through
// welford_save_state(mean_dst, group)/welford_restore_state(mean_dst, group)
// around each welford_update_rows, then welford_finalize_to_face per group.

#include <array>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

// Per-column working accumulator (the "LREG4/LREG5" of the SFPU): 32 columns.

// Saved per-group state for groupnorm's save/restore(group_id) swap. Silicon
// spills/fills the LREG accumulator to DST scratch per group; emule keeps the
// equivalent state in-memory keyed by group id.
static constexpr uint32_t __EMULE_WELFORD_MAX_GROUPS = 128;

inline void __emule_welford_clear() {
    std::memset(__emule_compute_ctx().welford_mean, 0, sizeof(__emule_compute_ctx().welford_mean));
    std::memset(__emule_compute_ctx().welford_m2, 0, sizeof(__emule_compute_ctx().welford_m2));
    std::memset(__emule_compute_ctx().welford_grp_mean, 0, sizeof(__emule_compute_ctx().welford_grp_mean));
    std::memset(__emule_compute_ctx().welford_grp_m2, 0, sizeof(__emule_compute_ctx().welford_grp_m2));
}

// Controls whether welford_init resets the running mean/M2 accumulators.
// Mirrors silicon `tt_metal/hw/inc/api/compute/welford.h`: ClearStats starts a
// fresh pass; PreserveStats re-arms the SFPU replay buffer mid-pass (e.g. after
// transpose_wh_tile clobbered the recurrence slots) without clearing LREG4/5.
enum class WelfordInitMode : uint8_t {
    ClearStats,
    PreserveStats,
};

// Templated to match silicon's signature. The template-with-default also covers
// the no-arg `welford_init()` callers (single non-templated overload would be
// ambiguous against it). emule has no SFPU replay buffer to reprogram, so the
// only observable effect is the conditional accumulator clear.
template <WelfordInitMode mode = WelfordInitMode::ClearStats>
ALWI void welford_init() {
    if constexpr (mode == WelfordInitMode::ClearStats) {
        __emule_welford_clear();
    }
}

// No-op in emule — silicon uses this to re-establish SFPU config; we have none.
ALWI void welford_reinit(uint32_t cbid, uint32_t call_line = __builtin_LINE()) {
    (void)cbid;
    (void)call_line;
}

ALWI void welford_clear() { __emule_welford_clear(); }

// Core update: fold rows [start_row, start_row+num_rows) of the input tile into
// the per-column working accumulator. The k-th processed row is sample number
// (start_idx + k + 1), so its Welford weight is 1/(start_idx + k + 1).
inline void __emule_welford_fold_rows(
    uint32_t input_dst_idx, uint32_t start_idx, uint32_t start_row, uint32_t num_rows) {
    __emule_dst_check(input_dst_idx, "welford_update");
    const float* tile = __emule_compute_ctx().dst[input_dst_idx];
    for (uint32_t k = 0; k < num_rows; ++k) {
        const uint32_t r = start_row + k;
        const float n = static_cast<float>(start_idx + k + 1);
        const float inv_n = 1.0f / n;
        const float* row = tile + r * 32;
        for (uint32_t c = 0; c < 32; ++c) {
            const float x = row[c];
            const float delta = x - __emule_compute_ctx().welford_mean[c];
            __emule_compute_ctx().welford_mean[c] += delta * inv_n;
            const float delta2 = x - __emule_compute_ctx().welford_mean[c];
            __emule_compute_ctx().welford_m2[c] += delta * delta2;
        }
    }
}

template <std::size_t reciprocal_size>
ALWI void welford_update(
    uint32_t input_dst_idx, uint32_t start_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    (void)reciprocal_lut;  // emule derives 1/n from start_idx directly
    __emule_welford_fold_rows(input_dst_idx, start_idx, 0, 32);
}

template <std::size_t reciprocal_size>
ALWI void welford_update_rows(
    uint32_t input_dst_idx,
    uint32_t start_idx,
    uint32_t start_row,
    uint32_t num_rows,
    const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    (void)reciprocal_lut;
    __emule_welford_fold_rows(input_dst_idx, start_idx, start_row, num_rows);
}

// ---- save/restore: swap the working accumulator with per-group state ----
// No-group overloads use group slot 0 (single accumulator, e.g. layernorm).
ALWI void welford_save_state(uint32_t mean_dst_idx, uint32_t group_id = 0) {
    (void)mean_dst_idx;
    const uint32_t g = (group_id < __EMULE_WELFORD_MAX_GROUPS) ? group_id : 0;
    std::memcpy(__emule_compute_ctx().welford_grp_mean[g], __emule_compute_ctx().welford_mean, sizeof(__emule_compute_ctx().welford_mean));
    std::memcpy(__emule_compute_ctx().welford_grp_m2[g], __emule_compute_ctx().welford_m2, sizeof(__emule_compute_ctx().welford_m2));
}

ALWI void welford_restore_state(uint32_t mean_dst_idx, uint32_t group_id = 0) {
    (void)mean_dst_idx;
    const uint32_t g = (group_id < __EMULE_WELFORD_MAX_GROUPS) ? group_id : 0;
    std::memcpy(__emule_compute_ctx().welford_mean, __emule_compute_ctx().welford_grp_mean[g], sizeof(__emule_compute_ctx().welford_mean));
    std::memcpy(__emule_compute_ctx().welford_m2, __emule_compute_ctx().welford_grp_m2[g], sizeof(__emule_compute_ctx().welford_m2));
}

// ---- finalize: write per-column mean and variance (M2 / N) into DST ----
// N = scale_idx + 1 (the caller passes the final sample count minus one, used to
// index the reciprocal LUT on silicon). Layernorm: scale_idx = W - 1 -> /W.
inline void __emule_welford_finalize(uint32_t mean_dst_idx, uint32_t scale_idx, bool to_face) {
    __emule_dst_check(mean_dst_idx, "welford_finalize.mean");
    __emule_dst_check(mean_dst_idx + 1, "welford_finalize.var");
    const float denom = static_cast<float>(scale_idx + 1);
    const float inv = 1.0f / denom;
    __emule_dst_mark_dirty(mean_dst_idx);
    __emule_dst_mark_dirty(mean_dst_idx + 1);
    std::memset(__emule_compute_ctx().dst[mean_dst_idx], 0, __EMULE_DST_BYTES);
    std::memset(__emule_compute_ctx().dst[mean_dst_idx + 1], 0, __EMULE_DST_BYTES);
    // In emule we write mean/variance into row 0 of their respective DST tiles.
    // (Silicon's *_to_face vs *_to_row differ in how the SFPU lays out lanes in DST.)
    (void)to_face;
    for (uint32_t c = 0; c < 32; ++c) {
        __emule_compute_ctx().dst[mean_dst_idx][c] = __emule_compute_ctx().welford_mean[c];
        __emule_compute_ctx().dst[mean_dst_idx + 1][c] = __emule_compute_ctx().welford_m2[c] * inv;
    }
}

template <std::size_t reciprocal_size>
ALWI void welford_finalize_to_row(
    uint32_t mean_dst_idx, uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    (void)reciprocal_lut;
    __emule_welford_finalize(mean_dst_idx, scale_idx, /*to_face=*/false);
}

template <std::size_t reciprocal_size>
ALWI void welford_finalize_to_face(
    uint32_t mean_dst_idx, uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    (void)reciprocal_lut;
    __emule_welford_finalize(mean_dst_idx, scale_idx, /*to_face=*/true);
}

// group_id finalize overload (groupnorm): restore group state then finalize.
template <std::size_t reciprocal_size>
ALWI void welford_finalize_to_face(
    uint32_t mean_dst_idx,
    uint32_t group_id,
    uint32_t scale_idx,
    const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    welford_restore_state(mean_dst_idx, group_id);
    welford_finalize_to_face<reciprocal_size>(mean_dst_idx, scale_idx, reciprocal_lut);
}

}  // namespace ckernel
