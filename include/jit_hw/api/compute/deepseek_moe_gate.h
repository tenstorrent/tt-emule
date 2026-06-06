// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Layer-1 (kernel-author tile-op API / PoR) shadow for the silicon
// Deepseek MoE gate compute kernel. The consumer op.hpp
// calls
//   deepseek_moe_gate_init<enable_sigmoid>(act, bias);
//   deepseek_moe_gate<enable_sigmoid>(act, bias, eps, scale);
// — same on silicon and emule. On silicon those resolve through the
// LLK / ckernel / TTI chain; on emule they resolve here.
//
// Contract (matches silicon):
//   - reads act and bias tiles from icb0 / icb1 (16×16 bf16, single-face)
//   - writes top-8 normalized scores into DST[0] (caller packs to out_scores CB)
//   - writes top-8 flat indices into DST[1] (caller packs to out_indices CB)
//   - input_indices CB is pre-loaded into DST[1] by the caller via
//     copy_tile, but we overwrite — silicon does the same.

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/bfloat16.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ckernel {

template <bool enable_sigmoid = false, bool is_32bit = false>
ALWI void deepseek_moe_gate_init(uint32_t /*icb0*/, uint32_t /*icb1*/) {
    // Silicon: programs SFPU LUTs, transpose addrmods, eltwise binary
    // configure_mop. Emule: no microarchitecture state to configure.
}

// Algorithm (matches the upstream Deepseek MoE gate golden):
//
//   Inputs are interpreted in the original (8, 32) layout — the host
//   reshapes (8, 32) → (16, 16) on the way to the device (act stays
//   linear; bias is transposed). The (16, 16) device-tile element at
//   (r_d, c_d) maps to (8, 32) position:
//       act_orig [r_d/2,      (r_d%2)*16 + c_d]
//       bias_orig[c_d/2,      (c_d%2)*16 + r_d]
//
//   1. scores  = sigmoid(act) if enable_sigmoid else act   (per element)
//   2. bias_scores = scores + bias                          (per element)
//   3. per (8,32) row: stable-sort cols by bias_score, desc
//   4. top2_sum per row = sorted_bias[r,0] + sorted_bias[r,1]
//   5. pick top-4 rows by top2_sum
//   6. candidates = sorted_bias[top4 rows] flattened → 4×32 = 128 values
//   7. pick top-8 from candidates by bias_score (carry the score+index)
//   8. normalize: out_score = score * scale / (sum(top8_scores) + eps)
//   9. write top-8 normalized scores to DST[0][0..7]
//      write top-8 flat indices (r*32+c) to DST[1][0..7] via bit-pack
//      (uint16(V) << 16) so pack_tile's bf16 truncation deposits the
//      raw uint16 bit pattern in L1 — matches the silicon LO16-only
//      indices encoding without needing a uint16 pack path.
template <bool enable_sigmoid = false, bool is_32bit = false>
ALWI void deepseek_moe_gate(uint32_t icb0, uint32_t icb1,
                            uint32_t eps_packed, uint32_t scale_packed) {
    constexpr uint32_t GR = 8;   // golden rows
    constexpr uint32_t GC = 32;  // golden cols

    const uint16_t* act_ptr = reinterpret_cast<const uint16_t*>(
        __emule_compute::cb_read_ptr_at(icb0, 0));
    const uint16_t* bias_ptr = reinterpret_cast<const uint16_t*>(
        __emule_compute::cb_read_ptr_at(icb1, 0));

    float scores[GR][GC];
    float bias_scores[GR][GC];
    for (uint32_t r_g = 0; r_g < GR; ++r_g) {
        for (uint32_t c_g = 0; c_g < GC; ++c_g) {
            uint32_t a_r = r_g * 2u + (c_g / 16u);
            uint32_t a_c = c_g % 16u;
            float s = __emule_bf16::to_f32(act_ptr[a_r * 16u + a_c]);
            if constexpr (enable_sigmoid) {
                s = 1.0f / (1.0f + std::exp(-s));
            }
            // Silicon stores DST as bf16 (DST_ACCUM_MODE=0); round-trip
            // through bf16 to match the precision at which downstream
            // top-K comparisons happen. Required to land seed-42
            // batch-1 cases where the 4th-best row is a near-tie.
            s = __emule_bf16::to_f32(__emule_bf16::from_f32(s));
            scores[r_g][c_g] = s;
            uint32_t b_r = c_g % 16u;
            uint32_t b_c = 2u * r_g + (c_g / 16u);
            float bs = s + __emule_bf16::to_f32(bias_ptr[b_r * 16u + b_c]);
            bias_scores[r_g][c_g] = __emule_bf16::to_f32(__emule_bf16::from_f32(bs));
        }
    }

    uint8_t row_perm[GR][GC];
    for (uint32_t r = 0; r < GR; ++r) {
        for (uint32_t c = 0; c < GC; ++c) row_perm[r][c] = static_cast<uint8_t>(c);
        std::stable_sort(&row_perm[r][0], &row_perm[r][0] + GC,
            [&](uint8_t a, uint8_t b) {
                return bias_scores[r][a] > bias_scores[r][b];
            });
    }

    float top2_sum[GR];
    for (uint32_t r = 0; r < GR; ++r) {
        top2_sum[r] = bias_scores[r][row_perm[r][0]] + bias_scores[r][row_perm[r][1]];
    }
    uint8_t top_rows[GR];
    for (uint32_t r = 0; r < GR; ++r) top_rows[r] = static_cast<uint8_t>(r);
    std::stable_sort(&top_rows[0], &top_rows[GR],
        [&](uint8_t a, uint8_t b) { return top2_sum[a] > top2_sum[b]; });

    constexpr uint32_t NCAND = 4 * GC;
    float cand_bias[NCAND], cand_score[NCAND];
    uint16_t cand_idx[NCAND];
    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t src_r = top_rows[i];
        for (uint32_t c = 0; c < GC; ++c) {
            uint32_t k = i * GC + c;
            uint8_t sc = row_perm[src_r][c];
            cand_bias[k] = bias_scores[src_r][sc];
            cand_score[k] = scores[src_r][sc];
            cand_idx[k] = static_cast<uint16_t>(src_r * GC + sc);
        }
    }

    uint16_t cand_perm[NCAND];
    for (uint32_t i = 0; i < NCAND; ++i) cand_perm[i] = static_cast<uint16_t>(i);
    std::stable_sort(&cand_perm[0], &cand_perm[NCAND],
        [&](uint16_t a, uint16_t b) { return cand_bias[a] > cand_bias[b]; });

    float top8_scores[8];
    uint16_t top8_indices[8];
    for (uint32_t i = 0; i < 8; ++i) {
        top8_scores[i] = cand_score[cand_perm[i]];
        top8_indices[i] = cand_idx[cand_perm[i]];
    }

    float eps_f, scale_f;
    std::memcpy(&eps_f, &eps_packed, sizeof(float));
    std::memcpy(&scale_f, &scale_packed, sizeof(float));
    float sum8 = 0.0f;
    for (uint32_t i = 0; i < 8; ++i) sum8 += top8_scores[i];
    float k = scale_f / (sum8 + eps_f);
    for (uint32_t i = 0; i < 8; ++i) top8_scores[i] *= k;

    // Write to DST. The caller will pack_tile DST[0]→out_scores (bf16)
    // and pack_tile DST[1]→out_indices (uint16 CB; bf16 truncation of
    // our bit-pack lands raw uint16 in L1).
    __emule_dst_check(0, "deepseek_moe_gate.scores");
    __emule_dst_check(1, "deepseek_moe_gate.indices");
    __emule_dst_mark_dirty(0);
    __emule_dst_mark_dirty(1);
    for (uint32_t i = 0; i < 8; ++i) {
        __emule_dst[0][i] = top8_scores[i];
        uint32_t idx_bits = static_cast<uint32_t>(top8_indices[i]) << 16;
        float idx_as_float;
        std::memcpy(&idx_as_float, &idx_bits, sizeof(float));
        __emule_dst[1][i] = idx_as_float;
    }
    for (uint32_t i = 8; i < __EMULE_TILE_ELEMS; ++i) {
        __emule_dst[0][i] = 0.0f;
        __emule_dst[1][i] = 0.0f;
    }
}

} // namespace ckernel
