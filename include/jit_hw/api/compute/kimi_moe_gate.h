// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Layer-1 (PoR) shadow for the silicon Kimi MoE gate compute kernel.
// Sibling of the Deepseek MoE gate + GLM MoE gate shadows. Same outline,
// different layout dance.
//
// Algorithm (matches the upstream Kimi MoE gate golden):
//
//   Inputs:
//     - act: padded (1, 1, 512) → (32, 16) on device. Linear: act_orig_flat[i] = act_dev[i/16][i%16].
//            Only first 384 indices are "valid"; positions ≥ 384 have bias=-inf so they're never selected.
//     - bias: padded (1, 1, 512) → reshape(2,16,16) → transpose(-2,-1) → reshape(32,16). So
//            bias_orig_flat[i] for i ∈ [0, 512) lives at device position
//            (face=i/256, r=c_d=(i%256)/16, c=r_d_lower=(i%256)%16) → bias_dev[face*16 + r_d_lower][c_d].
//
//   1. Compute bias_scores at every flat index i: act_orig_flat[i] + bias_orig_flat[i].
//   2. Flat top-8 by bias_score descending.
//   3. Gather scores from act_orig_flat (NOT bias_scores) at those indices.
//   4. Normalize: scores * scaling_factor / (sum(scores) + eps).
//   5. Write DST[0]=normalized scores, DST[1]=bit-packed indices.
//
// Note: the silicon kernel processes each face with a GLM-style top-8 sort
// then merges the two faces' top-8 via _kimi_moe_gate_merge_top8_. The
// emule shadow short-circuits that by computing flat top-8 across all 512
// elements directly — equivalent under the masked-bias regime the test uses.

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/bfloat16.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ckernel {

template <bool is_32bit = false>
ALWI void kimi_moe_gate_init(uint32_t /*icb0*/, uint32_t /*icb1*/) {
    // Silicon: programs unpacker variant + eltwise binary init + transpose
    // dest addrmods + topk SFPU init. Emule: no microarchitecture state.
}

template <bool normalize = false, int num_experts = 8, bool is_32bit = false>
ALWI void kimi_moe_gate(uint32_t icb0, uint32_t icb1,
                        uint32_t eps_packed, uint32_t scale_packed) {
    constexpr uint32_t TOTAL = 512;        // padded expert count
    constexpr uint32_t DEV_ROWS = 32;      // device tile rows
    constexpr uint32_t DEV_COLS = 16;      // device tile cols
    (void)DEV_ROWS; (void)DEV_COLS;

    const uint16_t* act_ptr = reinterpret_cast<const uint16_t*>(
        __emule_compute::cb_read_ptr_at(icb0, 0));
    const uint16_t* bias_ptr = reinterpret_cast<const uint16_t*>(
        __emule_compute::cb_read_ptr_at(icb1, 0));

    // bias_dev[r_d][c_d] for the (32, 16) device tile. r_d * 16 + c_d is
    // the device's linear offset (matches act_dev's linear order).
    auto bias_at = [&](uint32_t i) -> float {
        // Map original-flat i → device-flat offset for bias.
        uint32_t face = i / 256u;            // 0 or 1
        uint32_t r_d_lower = (i % 256u) % 16u;
        uint32_t c_d = (i % 256u) / 16u;
        uint32_t r_d = face * 16u + r_d_lower;
        return __emule_bf16::to_f32(bias_ptr[r_d * 16u + c_d]);
    };
    auto act_at = [&](uint32_t i) -> float {
        return __emule_bf16::to_f32(act_ptr[i]);  // linear order preserved
    };

    float scores[TOTAL];
    float bias_scores[TOTAL];
    for (uint32_t i = 0; i < TOTAL; ++i) {
        float s = act_at(i);
        s = __emule_bf16::to_f32(__emule_bf16::from_f32(s));
        float bs = s + bias_at(i);
        scores[i] = s;
        bias_scores[i] = __emule_bf16::to_f32(__emule_bf16::from_f32(bs));
    }

    // Flat top-K by bias_score descending. bias=-inf for invalid positions
    // (i ≥ 384) keeps them out of the top-K naturally.
    uint16_t perm[TOTAL];
    for (uint32_t i = 0; i < TOTAL; ++i) perm[i] = static_cast<uint16_t>(i);
    std::stable_sort(&perm[0], &perm[TOTAL],
        [&](uint16_t a, uint16_t b) { return bias_scores[a] > bias_scores[b]; });

    constexpr int K = num_experts;
    float topk_scores[K];
    uint16_t topk_indices[K];
    for (int i = 0; i < K; ++i) {
        topk_scores[i] = scores[perm[i]];
        topk_indices[i] = perm[i];
    }

    // Normalize. Kimi golden always normalizes (the `normalize` template
    // arg defaults true via the silicon-level Kimi MoE gate compute path
    // in practice, even when the template arg is false — the golden
    // unconditionally applies normalization).
    {
        float eps_f, scale_f;
        std::memcpy(&eps_f,   &eps_packed,   sizeof(float));
        std::memcpy(&scale_f, &scale_packed, sizeof(float));
        float sum_k = 0.0f;
        for (int i = 0; i < K; ++i) sum_k += topk_scores[i];
        float k = scale_f / (sum_k + eps_f);
        for (int i = 0; i < K; ++i) topk_scores[i] *= k;
    }

    // Write DST. DST[0] = scores; DST[1] = bit-packed indices.
    __emule_dst_check(0, "kimi_moe_gate.scores");
    __emule_dst_check(1, "kimi_moe_gate.indices");
    __emule_dst_mark_dirty(0);
    __emule_dst_mark_dirty(1);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        __emule_dst[0][i] = 0.0f;
        __emule_dst[1][i] = 0.0f;
    }
    for (int i = 0; i < K; ++i) {
        __emule_dst[0][i] = topk_scores[i];
        uint32_t idx_bits = static_cast<uint32_t>(topk_indices[i]) << 16;
        float idx_as_float;
        std::memcpy(&idx_as_float, &idx_bits, sizeof(float));
        __emule_dst[1][i] = idx_as_float;
    }
}

} // namespace ckernel
