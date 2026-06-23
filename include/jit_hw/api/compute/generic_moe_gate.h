// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Layer-1 (PoR) shadow for the silicon Generic MoE gate compute kernel.
// Sibling of the Deepseek / GLM / Kimi MoE gate shadows. generic_moe_gate is
// the parameterized flat top-K gate: top-`num_selected_experts` of
// `num_total_experts` by (act+bias), scores gathered from act, optional
// normalization, optional zero-padding of the unused output slots.
//
// Algorithm (matches generic_moe_gate_golden in test_generic_moe_gate.py):
//   flat_in[i]   = act[i],  flat_bias[i] = bias[i]   for i in [0, N)
//   bias_score[i]= flat_in[i] + flat_bias[i]
//   topk         = indices of the K largest bias_score (desc; inputs are
//                  distinct shuffled integers, so no ties)
//   scores       = flat_in[topk]            (gathered from act, NOT bias_score)
//   if normalize: scores = scores / (sum(scores) + eps) * scale
//   out: DST[0][0..K-1] = scores, DST[1][0..K-1] = topk flat indices.
//        The output is a (1,16) tile → pack reads row 0, cols 0..15
//        (DST[r*32+c] with r=0), so the K results live at DST[*][0..K-1].
//        Remaining slots are zeroed (satisfies the zero_tail contract; the
//        non-zero_tail case ignores the tail).
//
// Device-tile layout is linear for every supported N (128→(8,16), 256→(16,16),
// 384/512→(32,16)): each is row-major within its 16-wide face(s), so the CB
// page is read flat — same as the Kimi shadow.

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/bfloat16.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ckernel {

template <bool APPROX = false>
ALWI void generic_moe_gate_init(uint32_t /*icb0*/, uint32_t /*icb1*/) {
    // Silicon: eltwise-binary init + generic-moe-gate topk SFPU init.
    // Emule: no microarchitecture state to configure.
}

template <bool APPROX = false, bool normalize = false, int num_selected_experts = 8,
          int num_total_experts = 256, bool zero_tail = false>
ALWI void generic_moe_gate(uint32_t icb0, uint32_t icb1,
                           uint32_t eps_packed, uint32_t scale_packed) {
    constexpr uint32_t N = static_cast<uint32_t>(num_total_experts);
    constexpr int K = num_selected_experts;

    const uint16_t* act_ptr = reinterpret_cast<const uint16_t*>(
        __emule_compute::cb_read_ptr_at(icb0, 0));
    const uint16_t* bias_ptr = reinterpret_cast<const uint16_t*>(
        __emule_compute::cb_read_ptr_at(icb1, 0));

    float scores[N];
    float bias_scores[N];
    for (uint32_t i = 0; i < N; ++i) {
        float s = __emule_bf16::to_f32(act_ptr[i]);
        s = __emule_bf16::to_f32(__emule_bf16::from_f32(s));
        float bs = s + __emule_bf16::to_f32(bias_ptr[i]);
        scores[i] = s;
        bias_scores[i] = __emule_bf16::to_f32(__emule_bf16::from_f32(bs));
    }

    // Flat top-K by bias_score descending (ties → lower index, matching the
    // sorted torch.topk golden; the test uses distinct integers so ties don't
    // arise in practice).
    uint16_t perm[N];
    for (uint32_t i = 0; i < N; ++i) perm[i] = static_cast<uint16_t>(i);
    std::stable_sort(&perm[0], &perm[N],
        [&](uint16_t a, uint16_t b) { return bias_scores[a] > bias_scores[b]; });

    float topk_scores[K];
    uint16_t topk_indices[K];
    for (int i = 0; i < K; ++i) {
        topk_scores[i] = scores[perm[i]];
        topk_indices[i] = perm[i];
    }

    if constexpr (normalize) {
        float eps_f, scale_f;
        std::memcpy(&eps_f, &eps_packed, sizeof(float));
        std::memcpy(&scale_f, &scale_packed, sizeof(float));
        float sum_k = 0.0f;
        for (int i = 0; i < K; ++i) sum_k += topk_scores[i];
        float k = scale_f / (sum_k + eps_f);
        for (int i = 0; i < K; ++i) topk_scores[i] *= k;
    }

    // DST[0] = scores, DST[1] = bit-packed indices, at cols 0..K-1 of row 0.
    __emule_dst_check(0, "generic_moe_gate.scores");
    __emule_dst_check(1, "generic_moe_gate.indices");
    __emule_dst_mark_dirty(0);
    __emule_dst_mark_dirty(1);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        __emule_dst[0][i] = 0.0f;
        __emule_dst[1][i] = 0.0f;
    }
    // Outputs live at col 0 of rows 0..K-1 of face 0 (DST index i*32) — the
    // moe-gate pack-subrect contract shared with glm/kimi; the (1,16) output
    // tile is gathered from that column. Tail rows K..15 stay zeroed (above),
    // satisfying zero_tail.
    for (int i = 0; i < K; ++i) {
        __emule_dst[0][i * 32] = topk_scores[i];
        uint32_t idx_bits = static_cast<uint32_t>(topk_indices[i]);
        float idx_as_float;
        std::memcpy(&idx_as_float, &idx_bits, sizeof(float));
        __emule_dst[1][i * 32] = idx_as_float;
    }
}

}  // namespace ckernel
