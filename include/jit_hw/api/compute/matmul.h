// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT compute stub for matmul operations.
// matmul_tiles performs a 32x32 tile GEMM (bfloat16 inputs, float32 DST accumulate).
// Uses AVX2/FMA intrinsics when available for ~4-8x speedup over scalar.

#include "jit_hw/api/compute/common.h"
#include "jit_hw/internal/llk_state.h"
#include "jit_hw/llk/llk_reduce_primitives.h"
#include <utility>  // std::swap for the IN1 transpose path

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define EMULE_MATMUL_USE_AVX2 1
#endif

namespace ckernel {

// ---- Init stubs (hardware pipeline configuration) ----
// Signatures match silicon's tt_metal/hw/inc/api/compute/matmul.h: required
// CB ids and (for _with_dt) the old-srcA cb; transpose has a default of 0.
// `transpose=1` is honored: stored in `__llk_matmul_transpose` and applied
// to the IN1 tile in matmul_tiles below — see internal/llk_state.h.
ALWI void mm_init(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id,
                  uint32_t transpose = 0) {
    (void)in0_cb_id; (void)in1_cb_id; (void)out_cb_id;
    __emule_compute_ctx().llk_matmul_transpose = (transpose != 0);
}
ALWI void mm_init_short(uint32_t in0_cb_id, uint32_t in1_cb_id,
                        uint32_t transpose = 0) {
    (void)in0_cb_id; (void)in1_cb_id;
    __emule_compute_ctx().llk_matmul_transpose = (transpose != 0);
}
ALWI void mm_init_short_with_dt(uint32_t in0_cb_id, uint32_t in1_cb_id,
                                uint32_t c_in_old_srca, uint32_t transpose = 0) {
    (void)in0_cb_id; (void)in1_cb_id; (void)c_in_old_srca;
    __emule_compute_ctx().llk_matmul_transpose = (transpose != 0);
}

// #46346/#22219 renamed mm_init->matmul_init and mm_block_init->matmul_block_init
// (dropping out_cb_id, now owned by compute_kernel_hw_startup) + added a call_line
// default. Same emule body as the mm_* stubs: only transpose is load-bearing
// (applied to IN1 in matmul_tiles); CB ids / dims / call_line are signature parity.
ALWI void matmul_init(uint32_t in0_cb_id, uint32_t in1_cb_id,
                      const uint32_t transpose = 0,
                      uint32_t call_line = __builtin_LINE()) {
    (void)in0_cb_id; (void)in1_cb_id; (void)call_line;
    __emule_compute_ctx().llk_matmul_transpose = (transpose != 0);
}
ALWI void matmul_block_init(uint32_t in0_cb_id, uint32_t in1_cb_id,
                            const uint32_t transpose = 0, uint32_t ct_dim = 1,
                            uint32_t rt_dim = 1, uint32_t kt_dim = 1,
                            uint32_t call_line = __builtin_LINE()) {
    (void)in0_cb_id; (void)in1_cb_id;
    (void)ct_dim; (void)rt_dim; (void)kt_dim; (void)call_line;
    __emule_compute_ctx().llk_matmul_transpose = (transpose != 0);
}

// ---- matmul_tiles: tile GEMM accumulate into DST ----
// Reads tile A from CB[in0_cb] at tile offset in0_tile and tile B from
// CB[in1_cb] at tile offset in1_tile.  Accumulates A*B into DST[idst].
// DST stores float32 — acquire zeroes it, then matmul_tiles accumulates.
// Per-operand format dispatch via the central __emule_unpack_cb_tile_to
// (fp32 / bf16 / Bfp8_b / Bfp4_b) — silicon's unpacker reconfig is per-operand,
// so mixed-format inputs (e.g. bf16×Bfp4_b for MoE) decode correctly.
ALWI void matmul_tiles(uint32_t in0_cb, uint32_t in1_cb,
                       uint32_t in0_tile, uint32_t in1_tile, uint32_t idst) {
    __emule_dst_check(idst, "matmul_tiles");
    __emule_dst_mark_dirty(idst);
    // A is M×K, B is K×N (tiny-tile aware). DST and the operand buffers stay
    // 32-strided 32×32 grids with each tile in the top-left; only the M/K/N loop
    // bounds shrink. For a full 32×32 tile this is the standard square GEMM.
    constexpr uint32_t DIM = 32;
    const uint32_t M = get_tile_r_dim(in0_cb);
    const uint32_t K = get_tile_c_dim(in0_cb);
    const bool transpose = __emule_compute_ctx().llk_matmul_transpose;
    // Under transpose, in1 is stored as N×K and used as its transpose (K×N);
    // otherwise in1 is B directly (K×N).
    const uint32_t N = transpose ? get_tile_r_dim(in1_cb) : get_tile_c_dim(in1_cb);
    float a_rm[DIM * DIM];
    float b_rm[DIM * DIM];
    // UNPACK both operands via the central format-aware reader (nfaces→row-major).
    // Decodes fp32 / bf16 / Bfp8_b / Bfp4_b independently per operand, so mixed
    // formats (e.g. bf16 activations × Bfp4_b weights, as in MoE) decode correctly.
    __emule_unpack_cb_tile_to(in0_cb, in0_tile, a_rm);
    __emule_unpack_cb_tile_to(in1_cb, in1_tile, b_rm);
    // Apply IN1 transpose if mm_init(... transpose=1) was set — silicon does this
    // in the unpacker; emule transposes the decoded N×K view into b_t (K×N).
    float b_t[DIM * DIM];
    const float* b = b_rm;
    if (transpose) {
        std::memset(b_t, 0, sizeof(b_t));  // zero inactive lanes (AVX2 reads 8-wide)
        const uint32_t h_b = get_tile_r_dim(in1_cb);  // stored rows (= N)
        const uint32_t w_b = get_tile_c_dim(in1_cb);  // stored cols (= K)
        for (uint32_t r = 0; r < h_b; r++)
            for (uint32_t c = 0; c < w_b; c++)
                b_t[c * DIM + r] = b_rm[r * DIM + c];
        b = b_t;
    }
    // MATH: DST[m,n] += A[m,k] * B[k,n], accumulating into the 32-strided DST.
    // Matmul-backed reductions use zero B lanes as masks for padded positions.
    // Skip those lanes so log/reciprocal reductions do not turn inf * 0 into NaN.
#ifdef EMULE_MATMUL_USE_AVX2
    const __m256 zero_vec = _mm256_setzero_ps();
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t k = 0; k < K; k++) {
            __m256 a_vec = _mm256_set1_ps(a_rm[m * DIM + k]);
            for (uint32_t n = 0; n < N; n += 8) {  // N ∈ {16,32}; inactive b lanes are 0
                __m256 b_vec = _mm256_loadu_ps(&b[k * DIM + n]);
                __m256 b_zero = _mm256_cmp_ps(b_vec, zero_vec, _CMP_EQ_OQ);
                __m256 a_masked = _mm256_blendv_ps(a_vec, zero_vec, b_zero);
                __m256 d_vec = _mm256_loadu_ps(&__emule_compute_ctx().dst[idst][m * DIM + n]);
                d_vec = _mm256_fmadd_ps(a_masked, b_vec, d_vec);
                _mm256_storeu_ps(&__emule_compute_ctx().dst[idst][m * DIM + n], d_vec);
            }
        }
    }
#else
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t k = 0; k < K; k++) {
            float a_val = a_rm[m * DIM + k];
            for (uint32_t n = 0; n < N; n++) {
                const float b_val = b[k * DIM + n];
                if (b_val != 0.0f) {
                    __emule_compute_ctx().dst[idst][m * DIM + n] += a_val * b_val;
                }
            }
        }
    }
#endif
}

// ---- Block matmul stubs (not used by bmm.cpp simple path) ----
// Signatures match silicon's mm_block_init / mm_block_init_short[_with_dt] /
// matmul_block: ct/rt/kt_dim have defaults of 1, transpose has default 0,
// CB ids are required.
ALWI void mm_block_init(uint32_t in0_cb_id, uint32_t in1_cb_id,
                        uint32_t out_cb_id, uint32_t transpose = 0,
                        uint32_t ct_dim = 1, uint32_t rt_dim = 1,
                        uint32_t kt_dim = 1) {
    (void)in0_cb_id; (void)in1_cb_id; (void)out_cb_id;
    (void)ct_dim; (void)rt_dim; (void)kt_dim;
    __emule_compute_ctx().llk_matmul_transpose = (transpose != 0);
}
ALWI void mm_block_init_short(uint32_t in0_cb_id, uint32_t in1_cb_id,
                              uint32_t transpose = 0, uint32_t ct_dim = 1,
                              uint32_t rt_dim = 1, uint32_t kt_dim = 1) {
    (void)in0_cb_id; (void)in1_cb_id;
    (void)ct_dim; (void)rt_dim; (void)kt_dim;
    __emule_compute_ctx().llk_matmul_transpose = (transpose != 0);
}
ALWI void mm_block_init_short_with_dt(uint32_t in0_cb_id, uint32_t in1_cb_id,
                                      uint32_t old_in1_cb_id, uint32_t transpose = 0,
                                      uint32_t ct_dim = 1, uint32_t rt_dim = 1,
                                      uint32_t kt_dim = 1) {
    (void)in0_cb_id; (void)in1_cb_id; (void)old_in1_cb_id;
    (void)ct_dim; (void)rt_dim; (void)kt_dim;
    __emule_compute_ctx().llk_matmul_transpose = (transpose != 0);
}
// _with_both_dt reconfigures both input CB formats; emule reads CB format
// per-call so the reconfig is implicit (only the transpose state matters).
ALWI void mm_block_init_short_with_both_dt(uint32_t in0_cb_id, uint32_t in1_cb_id,
                                           uint32_t old_in0_cb_id, uint32_t old_in1_cb_id,
                                           uint32_t transpose = 0, uint32_t ct_dim = 1,
                                           uint32_t rt_dim = 1, uint32_t kt_dim = 1) {
    (void)in0_cb_id; (void)in1_cb_id; (void)old_in0_cb_id; (void)old_in1_cb_id;
    (void)ct_dim; (void)rt_dim; (void)kt_dim;
    __emule_compute_ctx().llk_matmul_transpose = (transpose != 0);
}
// matmul_block: compute rt_dim × ct_dim block of output tiles.
// For each output tile (r, c): DST[idst + r*ct_dim + c] += A[in0_tile + r*kt_dim] * B[in1_tile + c]
// The runtime `transpose` arg here overrides any value previously set by
// mm_block_init for this block call. All args required, matching silicon.
ALWI void matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id,
                       uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst,
                       uint32_t transpose, uint32_t ct_dim,
                       uint32_t rt_dim, uint32_t kt_dim) {
    if (rt_dim * ct_dim > 0)
        __emule_dst_check(idst + rt_dim * ct_dim - 1, "matmul_block");
    __emule_compute_ctx().llk_matmul_transpose = (transpose != 0);
    uint32_t dst = idst;
    for (uint32_t r = 0; r < rt_dim; r++) {
        for (uint32_t c = 0; c < ct_dim; c++) {
            matmul_tiles(in0_cb_id, in1_cb_id,
                         in0_tile_index + r * kt_dim,
                         in1_tile_index + c,
                         dst);
            dst++;
        }
    }
}

} // namespace ckernel

// ---- Preempt tt-mlir verbatim injection of experimental_matmul_llks ----
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MATMUL_LLKS_H

namespace experimental {
ALWI void matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id,
                       uint32_t in0_tile_index, uint32_t in1_tile_index,
                       uint32_t idst, const uint32_t transpose, uint32_t ct_dim,
                       uint32_t rt_dim, uint32_t kt_dim, uint32_t nt_dim) {
    if (rt_dim * ct_dim > 0)
        __emule_dst_check(idst + rt_dim * ct_dim - 1, "experimental::matmul_block");
    for (uint32_t i = 0; i < kt_dim; i++) {
        ckernel::matmul_block(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index,
                              idst, transpose, ct_dim, rt_dim, kt_dim);
        in0_tile_index++;
        in1_tile_index += nt_dim;
    }
}
} // namespace experimental
