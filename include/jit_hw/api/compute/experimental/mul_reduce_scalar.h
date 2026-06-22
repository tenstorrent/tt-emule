// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule model + shadow for the fused multiply-reduce-scalar compute op.
//
// Net semantics: dst[0][0] = scaler^2 * sum(A*B) over all elements of
// `num_tiles` tiles. The fused multiply-reduce multiplies by the srcB scalar at
// BOTH the column-reduce and the final scalar-reduce stages, so the scaler is
// squared (the documented sum(x^2 * scalar^2) behaviour; callers fold a mean's
// 1/N by passing scalar = 1/sqrt(N)).
//
// blaze's sum_of_squares kernel INLINES the raw LLK state machine (multiply
// phase → switch-to-reduce → per-tile column reduce → final scalar reduce)
// rather than calling the high-level mul_reduce_scalar_tile wrapper, so this
// shadow provides every LLK entry point the kernel touches, not just the
// wrapper. Each is modelled against the net effect using thread-local "register"
// state; the real per-TRISC SFPU/FPU LLKs (pulled by upstream's
// TRISC_MATH/TRISC_UNPACK includes) are not consumable by emule. The multiply
// step reads both operands the paired llk_unpack_AB selected (via
// __emule_unpack_AB_state), so it models a genuine A*B, not a hardcoded square.

#include "jit_hw/api/compute/common.h"              // pulls llk_unpack_a.h → __emule_unpack_AB_state
#include "jit_hw/api/compute/eltwise_unary/fill.h"  // fill_tile

#include <cstdint>

namespace ckernel {

namespace __emule_mrs {
inline thread_local float srcA[__EMULE_TILE_ELEMS];
inline thread_local double acc = 0.0;     // running scaler*sum accumulator
inline thread_local float scaler = 1.0f;  // srcB broadcast value
}  // namespace __emule_mrs

// ── high-level wrapper init/uninit ───────────────────────────────────────
ALWI void mul_reduce_scalar_init(uint32_t /*icb0*/, uint32_t /*icb1*/) {}
ALWI void mul_reduce_scalar_uninit() {}

template <MathFidelity /*mf*/ = MathFidelity::HiFi4>
ALWI void llk_math_eltwise_mul_reduce_scalar_init(uint32_t /*operand_A*/, uint32_t /*acc_to_dest*/ = 0) {}

// Multiply phase: dst[dst_index] = A * B, where A/B are the operand tiles the
// paired llk_unpack_AB selected (icb0 here is the math-side operand-A handle,
// matching the unpacker's operandA). For sum_of_squares A == B == input.
template <bool is_fp32_dest_acc_en = false, MathFidelity math_fidelity = MathFidelity::HiFi4>
ALWI void llk_math_eltwise_mul_reduce_scalar(uint32_t dst_index, uint32_t /*icb0*/, bool /*clear*/ = true) {
    __emule_dst_check(dst_index, "llk_math_eltwise_mul_reduce_scalar");
    const auto& ab = __emule_unpack_AB_state;
    float a[__EMULE_TILE_ELEMS];
    float b[__EMULE_TILE_ELEMS];
    __emule_unpack_cb_tile_to(ab.operandA, ab.tile_index_a, a);
    __emule_unpack_cb_tile_to(ab.operandB, ab.tile_index_b, b);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_dst[dst_index][i] = a[i] * b[i];
    }
}

ALWI void llk_unpack_mul_reduce_scalar_switch_to_reduce() {}

template <bool is_fp32_dest_acc_en = false, MathFidelity math_fidelity = MathFidelity::HiFi4,
          bool enforce_fp32_accumulation = false>
ALWI void llk_math_mul_reduce_scalar_reduce_init() {
    __emule_mrs::acc = 0.0;
    __emule_mrs::scaler = 1.0f;
}

template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void llk_math_mul_reduce_scalar_move_dest_to_src(uint32_t idst = 0) {
    __emule_dst_check(idst, "llk_math_mul_reduce_scalar_move_dest_to_src");
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA) {
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) __emule_mrs::srcA[i] = __emule_dst[idst][i];
    } else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB) {
        // srcB is the scalar broadcast (filled uniformly): capture element 0.
        __emule_mrs::scaler = __emule_dst[idst][0];
    }
}

// _calculate_fill_(dst_idx, value): the fill SFPU op, modelled by reusing
// fill_tile (eltwise_unary/fill.h). Co-defined with the macro below so the
// kernel's SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM(_calculate_fill_, …, DST_IDX,
// PARAM0) routes here without the SFPU vector-call machinery. (emule's
// _calculate_fill_ takes the DST index explicitly where silicon's takes it
// implicitly via the current-dst register.)
ALWI void _calculate_fill_(uint32_t dst_idx, float value) { fill_tile(dst_idx, value); }

#ifndef SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM
#define SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM(FN, MODE, APPROXIMATE, EXTRA_PARAM, DST_IDX, PARAM0) \
    FN(DST_IDX, PARAM0)
#endif

template <bool untilize = false, ReduceDim reduce_dim = ReduceDim::REDUCE_SCALAR>
ALWI void llk_pack_reduce_mask_config() {}
ALWI void llk_pack_reduce_mask_clear() {}

// Column reduce: accumulate scaler * sum(srcA) into the running scalar.
template <MathFidelity math_fidelity = MathFidelity::HiFi4>
ALWI void llk_math_mul_reduce_column(uint32_t /*dst_index*/, uint32_t /*icb0*/) {
    double s = 0.0;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) s += static_cast<double>(__emule_mrs::srcA[i]);
    __emule_mrs::acc += static_cast<double>(__emule_mrs::scaler) * s;
}

// Final scalar reduce: deposit the result at DST[0][0] (reduce-scalar
// convention: result lives at face 0, element [0,0]). srcB (the scaler) is
// applied a SECOND time here — the fused multiply-reduce multiplies by srcB at
// both the column-reduce and the final scalar-reduce stages, so the net scaling
// is scaler^2 (the documented sum(x^2 * scalar^2) semantics; the caller passes
// scalar = 1/sqrt(N) to fold in the mean's 1/N).
template <MathFidelity math_fidelity = MathFidelity::HiFi4>
ALWI void llk_math_mul_reduce_scalar() {
    __emule_dst_check(0, "llk_math_mul_reduce_scalar");
    __emule_dst[0][0] = static_cast<float>(__emule_mrs::acc * static_cast<double>(__emule_mrs::scaler));
}

ALWI void llk_math_mul_reduce_scalar_clear_dvalid() {}

// High-level wrapper (sum(A*B over num_tiles) * scaler → dst[0][0]); provided
// for API completeness — handles the general icb0 != icb1 case the inlined
// path above cannot, by reading both operands directly.
template <PoolType reduce_type = PoolType::SUM>
ALWI void mul_reduce_scalar_tile(uint32_t icb0, uint32_t icb1, uint32_t num_tiles, float scaler = 1.0f) {
    __emule_dst_check(0, "mul_reduce_scalar_tile");
    float a[__EMULE_TILE_ELEMS];
    float b[__EMULE_TILE_ELEMS];
    double sum = 0.0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        __emule_unpack_cb_tile_to(icb0, t, a);
        __emule_unpack_cb_tile_to(icb1, t, b);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) sum += static_cast<double>(a[i]) * b[i];
    }
    // scaler applied twice (column-reduce + scalar-reduce both multiply by srcB),
    // matching the inlined-LLK path's sum(A*B * scaler^2) net effect.
    __emule_dst[0][0] = static_cast<float>(sum * static_cast<double>(scaler) * scaler);
}

}  // namespace ckernel
