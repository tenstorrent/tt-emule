// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for the MatmulFusedAct kernel body.
// Provides:
//   - ckernel::custom_mm_block_init_short<>, custom_mm_block<>, custom_mm_block_uninit<>
//     (route to existing emule matmul_block GEMM accumulate path)
//   - ckernel::packer::get_packer_dest_offset() (no-op host stub)
//   - ckernel::llk_math_eltwise_unary_sfpu_silu_init/_silu<> templates
//     (route to ckernel::silu_tile / silu_tile_init)
//   - ckernel::llk_math_eltwise_unary_sfpu_sigmoid_init/_sigmoid<> templates
//     (route to ckernel::sigmoid_tile / sigmoid_tile_init)
//   - TTI_SEMWAIT / TTI_STALLWAIT / TT_SETC16 macro stubs
//     (silicon SFPU pipeline barriers — no-ops in software emulation)
//   - DEST_TARGET_REG_CFG_MATH_Offset_ADDR32 constant
//   - ckernel::semaphore::t6_sem() helper

#include <cstdint>

#include "api/compute/matmul.h"  // pulls ckernel::matmul_block

namespace ckernel {

// TTI_SEMWAIT / TTI_STALLWAIT / TT_SETC16 (no-ops in emule) come from common.h
// via the matmul.h include above — single definition site, not redefined here.

// ---- Silicon firmware constants (referenced by TT_SETC16 sites) ----
#ifndef DEST_TARGET_REG_CFG_MATH_Offset_ADDR32
constexpr uint32_t DEST_TARGET_REG_CFG_MATH_Offset_ADDR32 = 0;
#endif

// ---- Packer helper ----
namespace packer {
ALWI uint32_t get_packer_dest_offset() { return 0; }
}  // namespace packer

// ---- Semaphore helper ----
// On silicon, semaphore::t6_sem(MATH_PACK) returns a register address;
// here it's used as an opaque token argument to TTI_SEMWAIT (no-op).
namespace semaphore {
ALWI uint32_t t6_sem(uint32_t sem_id) { (void)sem_id; return 0; }
}  // namespace semaphore

// ---- custom_mm_block_* templates ----
// Silicon: highly optimized matmul-with-pack. Emule: delegate to
// matmul_block (already implements 32x32 tile GEMM accumulate into DST).
//
// Signature based on usage:
//   custom_mm_block_init_short<transpose, split_acc, dense_packing>(in0_cb, in1_cb, out_cb, ct_dim)
//   custom_mm_block<finalize, read_transposed>(in0_cb, in1_cb, in0_tile, in1_tile, idst, kt_dim)
//   custom_mm_block_uninit<dense_packing>()

template <bool transpose = false, bool split_acc = false, bool dense_packing = false>
ALWI void custom_mm_block_init_short(uint32_t in0_cb, uint32_t in1_cb,
                                     uint32_t out_cb, uint32_t ct_dim = 1) {
    (void)in0_cb; (void)in1_cb; (void)out_cb; (void)ct_dim;
}

// Full init variant — same emule no-op as the _short form.
template <bool transpose = false, bool split_acc = false,
          bool dense_packing = false, bool fp32_dest_acc_en = false>
ALWI void custom_mm_block_init(uint32_t in0_cb, uint32_t in1_cb,
                               uint32_t out_cb, uint32_t ct_dim = 1) {
    (void)in0_cb; (void)in1_cb; (void)out_cb; (void)ct_dim;
}

template <bool finalize = false, bool read_transposed = false>
ALWI void custom_mm_block(uint32_t in0_cb, uint32_t in1_cb,
                          uint32_t in0_tile, uint32_t in1_tile,
                          uint32_t idst, uint32_t kt_dim,
                          uint32_t ct_dim = 1) {
    // Accumulate kt_dim partial products across ct_dim output tiles:
    //   for c in [0, ct_dim): DST[idst+c] += sum_k A[in0_tile+k] * B[in1_tile + k*ct_dim + c]
    // For 32x32 tiles. in0 is [1×kt], in1 is [kt×ct].
    if (ct_dim > 0)
        __emule_dst_check(idst + ct_dim - 1, "custom_mm_block");
    for (uint32_t c = 0; c < ct_dim; ++c) {
        for (uint32_t k = 0; k < kt_dim; ++k) {
            matmul_tiles(in0_cb, in1_cb,
                         in0_tile + k,
                         in1_tile + k * ct_dim + c,
                         idst + c);
        }
    }
}

template <bool dense_packing = false>
ALWI void custom_mm_block_uninit() {}

// ---- LLK silu/sigmoid init+apply (lowered from upstream TRISC_PACK path) ----
template <bool approx = false>
ALWI void llk_math_eltwise_unary_sfpu_silu_init() {}

template <bool approx = false, bool acc = false, int iter = 2>
ALWI void llk_math_eltwise_unary_sfpu_silu(uint32_t idst, int vector_mode) {
    (void)vector_mode;
    __emule_dst_check(idst, "llk_math_eltwise_unary_sfpu_silu");
    float* d = __emule_compute_ctx().dst[idst];
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        float x = d[i];
        d[i] = x / (1.0f + std::exp(-x));
    }
}

template <bool approx = false>
ALWI void llk_math_eltwise_unary_sfpu_sigmoid_init() {}

template <bool approx = false, bool acc = false, int iter = 2>
ALWI void llk_math_eltwise_unary_sfpu_sigmoid(uint32_t idst, int vector_mode) {
    (void)vector_mode;
    __emule_dst_check(idst, "llk_math_eltwise_unary_sfpu_sigmoid");
    float* d = __emule_compute_ctx().dst[idst];
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        d[i] = 1.0f / (1.0f + std::exp(-d[i]));
    }
}

}  // namespace ckernel
