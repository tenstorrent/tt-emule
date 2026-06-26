// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule model for the H128 (1x128) Walsh-Hadamard transform compute op.
// Replaces blaze's compute_kernel_api/hadamard.h, whose hadamard_h128_*
// bodies expand to the custom narrow-MOP LLKs (llk_unpack_hadamard_h128* /
// llk_math_hadamard_h128*) that emule can't consume. Computes the net
// transform directly: Y = H_128 @ x, and Y /= sqrt(128) when normalize.
//
// Layout (locked against tests/blaze/micro_ops/test_hadamard_h128.py): the
// 1x128 input occupies face-0 rows [0..8) x cols [0..16) of in_cb — x[k] at
// (row k/16, col k%16). The result goes to the same face-0 rows [0..8) x
// cols [0..16) of DST — Y[j] at __emule_dst[idst][(j/16)*32 + (j%16)] (the
// 32-strided row-major DST grid; the test reads out[:8,:16].reshape(-1)).
// h16_cb (the silicon H_16 building block) and MATH_FIDELITY are silicon-
// pipeline details with no effect on the net result, so the model ignores
// them. Sylvester Hadamard closed form: H_128[j][k] = (-1)^popcount(j & k).

#include "api/compute/common.h"
#include "api/compute/reconfig_data_format.h"

#include <cmath>
#include <cstdint>

namespace ckernel {

template <bool normalize = true, bool fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hadamard_h128_init(uint32_t /*in_cb_id*/, uint32_t /*h16_cb_id*/,
                             uint32_t out_cb_id) {
    // Silicon programs the custom unpack/math addrmods here; emule only needs
    // the packer pointed at the bfp8 output CB (global pack_init(0) selected
    // CB0's bf16 geometry) — matches blaze's hadamard_h128_init.
    pack_reconfig_data_format<true>(out_cb_id);
}

template <bool normalize = true>
ALWI void hadamard_h128_tile(uint32_t in_cb_id, uint32_t /*h16_cb_id*/,
                             uint32_t in_tile_index, uint32_t /*h16_tile_index*/,
                             uint32_t dst_index) {
    __emule_dst_check(dst_index, "hadamard_h128_tile");
    __emule_dst_mark_dirty(dst_index);

    float in_rm[__EMULE_TILE_ELEMS];
    __emule_unpack_cb_tile_to(in_cb_id, in_tile_index, in_rm);

    // Gather the 1x128 input: x[k] = face-0 (row k/16, col k%16).
    float x[128];
    for (uint32_t k = 0; k < 128u; ++k) {
        x[k] = in_rm[(k >> 4) * 32u + (k & 15u)];
    }

    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        __emule_dst[dst_index][i] = 0.0f;
    }

    // Y[j] = sum_k H_128[j][k] * x[k]; H_128[j][k] = (-1)^popcount(j & k).
    constexpr float kInvSqrt128 = 0.08838834764831845f;  // 1/sqrt(128)
    for (uint32_t j = 0; j < 128u; ++j) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < 128u; ++k) {
            acc += (static_cast<uint32_t>(__builtin_popcount(j & k)) & 1u) ? -x[k] : x[k];
        }
        if constexpr (normalize) {
            acc *= kInvSqrt128;
        }
        __emule_dst[dst_index][(j >> 4) * 32u + (j & 15u)] = acc;
    }
}

inline void hadamard_h128_uninit() {}

}  // namespace ckernel
