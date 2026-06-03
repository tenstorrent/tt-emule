// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/welford.h`. Welford's online mean/variance.
// Thread-local 1024-lane mean/M2 + sample count mirror silicon LREG4/LREG5;
// save_state/restore_state round-trip them through a DST tile pair.
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_welfords.h

#include <array>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

static thread_local float __emule_welford_mean[__EMULE_TILE_ELEMS] = {};
static thread_local float __emule_welford_m2[__EMULE_TILE_ELEMS] = {};
static thread_local uint32_t __emule_welford_count = 0;

inline void __emule_welford_clear() {
    std::memset(__emule_welford_mean, 0, sizeof(__emule_welford_mean));
    std::memset(__emule_welford_m2, 0, sizeof(__emule_welford_m2));
    __emule_welford_count = 0;
}

ALWI void welford_init() { __emule_welford_clear(); }

// No-op in emule — silicon uses this to re-establish SFPU config; we have none.
ALWI void welford_reinit(uint32_t cbid, uint32_t call_line = __builtin_LINE()) {
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
    // Partial tile: only rows [start_row, start_row + num_rows) update. Count
    // still advances once per call (one sample per active lane).
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

// Spill mean to DST[mean_dst_idx] and M2 to DST[mean_dst_idx + 1].
ALWI void welford_save_state(uint32_t mean_dst_idx) {
    __emule_dst_check(mean_dst_idx, "welford_save_state");
    __emule_dst_check(mean_dst_idx + 1, "welford_save_state");
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
    // Store column-0 stats in row 0; zero the rest (LLK contract: only row 0
    // is valid). M2/n yields the variance.
    const float denom = (__emule_welford_count > 0) ? static_cast<float>(__emule_welford_count) : 1.0f;
    std::memset(__emule_dst[mean_dst_idx], 0, __EMULE_DST_BYTES);
    std::memset(__emule_dst[mean_dst_idx + 1], 0, __EMULE_DST_BYTES);
    for (uint32_t c = 0; c < 32; ++c) {
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
    // Lay out mean (row 0) / variance (row 1) in the first face; zero the rest.
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

// group_id overloads — emule keeps a single accumulator; the id is ignored.
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
