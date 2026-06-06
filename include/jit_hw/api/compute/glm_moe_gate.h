// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Layer-1 (PoR) shadow for the silicon GLM MoE gate compute kernel.
// Same pattern as the Deepseek MoE gate shadow.
//
// Algorithm (matches the upstream GLM MoE gate golden):
//
//   1. Reconstruct (8, 32) view from (16, 16) device tile (host reshape,
//      bias is pre-transposed) — identical layout dance to deepseek.
//   2. bias_scores = act + bias (per element).
//   3. Flat top-K across all 256 bias_scores (no row grouping like deepseek;
//      torch.topk on the flattened tensor).
//   4. Gather scores from `act` (NOT bias_scores) at the top-K indices.
//   5. If normalize: scores = scores * scale / (sum(scores) + eps).
//   6. Write top-K (normalized) scores to DST[0], bit-packed indices to DST[1].
//      If !normalize, also write top-K bias_scores to DST[2] (for the
//      out_bias_scores CB).

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/bfloat16.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ckernel {

template <bool is_32bit = false>
ALWI void glm_moe_gate_init(uint32_t /*icb0*/, uint32_t /*icb1*/) {
    // Silicon: programs eltwise binary init + transpose dest addrmods + topk
    // SFPU init. Emule: no microarchitecture state to configure.
}

template <bool normalize, int num_experts, bool is_32bit = false>
ALWI void glm_moe_gate(uint32_t icb0, uint32_t icb1,
                       uint32_t eps_packed, uint32_t scale_packed) {
    constexpr uint32_t GR = 8;   // golden rows (original (8,32))
    constexpr uint32_t GC = 32;  // golden cols
    constexpr uint32_t N = GR * GC;  // 256

    const uint16_t* act_ptr = reinterpret_cast<const uint16_t*>(
        __emule_compute::cb_read_ptr_at(icb0, 0));
    const uint16_t* bias_ptr = reinterpret_cast<const uint16_t*>(
        __emule_compute::cb_read_ptr_at(icb1, 0));

    // Rebuild (8, 32) views from the (16, 16) device tile. Identical layout
    // mapping to deepseek_moe_gate.h.
    float scores[N];        // scores from act (no sigmoid for GLM)
    float bias_scores[N];
    for (uint32_t r_g = 0; r_g < GR; ++r_g) {
        for (uint32_t c_g = 0; c_g < GC; ++c_g) {
            uint32_t a_r = r_g * 2u + (c_g / 16u);
            uint32_t a_c = c_g % 16u;
            float s = __emule_bf16::to_f32(act_ptr[a_r * 16u + a_c]);
            s = __emule_bf16::to_f32(__emule_bf16::from_f32(s));
            uint32_t b_r = c_g % 16u;
            uint32_t b_c = 2u * r_g + (c_g / 16u);
            float bs = s + __emule_bf16::to_f32(bias_ptr[b_r * 16u + b_c]);
            const uint32_t k = r_g * GC + c_g;
            scores[k] = s;
            bias_scores[k] = __emule_bf16::to_f32(__emule_bf16::from_f32(bs));
        }
    }

    // Flat top-K by bias_score descending; carry the (8,32) flat index.
    uint16_t perm[N];
    for (uint32_t i = 0; i < N; ++i) perm[i] = static_cast<uint16_t>(i);
    std::stable_sort(&perm[0], &perm[N],
        [&](uint16_t a, uint16_t b) { return bias_scores[a] > bias_scores[b]; });

    float topk_scores[num_experts];
    float topk_bias_scores[num_experts];
    uint16_t topk_indices[num_experts];
    for (int i = 0; i < num_experts; ++i) {
        topk_scores[i] = scores[perm[i]];
        topk_bias_scores[i] = bias_scores[perm[i]];
        topk_indices[i] = perm[i];
    }

    if constexpr (normalize) {
        float eps_f, scale_f;
        std::memcpy(&eps_f,   &eps_packed,   sizeof(float));
        std::memcpy(&scale_f, &scale_packed, sizeof(float));
        float sum_k = 0.0f;
        for (int i = 0; i < num_experts; ++i) sum_k += topk_scores[i];
        float k = scale_f / (sum_k + eps_f);
        for (int i = 0; i < num_experts; ++i) topk_scores[i] *= k;
    }

    // Write DST. DST[0] = scores; DST[1] = bit-packed indices; DST[2] =
    // top-K bias_scores when !normalize (silicon emits these via pack_tile(2)).
    __emule_dst_check(0, "glm_moe_gate.scores");
    __emule_dst_check(1, "glm_moe_gate.indices");
    __emule_dst_mark_dirty(0);
    __emule_dst_mark_dirty(1);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        __emule_dst[0][i] = 0.0f;
        __emule_dst[1][i] = 0.0f;
    }
    for (int i = 0; i < num_experts; ++i) {
        __emule_dst[0][i] = topk_scores[i];
        uint32_t idx_bits = static_cast<uint32_t>(topk_indices[i]) << 16;
        float idx_as_float;
        std::memcpy(&idx_as_float, &idx_bits, sizeof(float));
        __emule_dst[1][i] = idx_as_float;
    }
    if constexpr (!normalize) {
        __emule_dst_check(2, "glm_moe_gate.bias_scores");
        __emule_dst_mark_dirty(2);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) __emule_dst[2][i] = 0.0f;
        for (int i = 0; i < num_experts; ++i) __emule_dst[2][i] = topk_bias_scores[i];
    }
}

} // namespace ckernel
