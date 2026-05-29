// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT compute stub for matmul operations.
// matmul_tiles performs a 32x32 tile GEMM (bfloat16 inputs, float32 DST accumulate).
// Uses AVX2/FMA intrinsics when available for ~4-8x speedup over scalar.

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/nfaces.h"
#include "jit_hw/llk/llk_reduce_primitives.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define EMULE_MATMUL_USE_AVX2 1
#endif

namespace ckernel {

// ---- Init stubs (hardware pipeline configuration) ----
ALWI void mm_init(uint32_t in0_cb = 0, uint32_t in1_cb = 1, uint32_t out_cb = 16,
                  uint32_t transpose = 0) {}
ALWI void mm_init_short(uint32_t in0_cb = 0, uint32_t in1_cb = 1,
                        uint32_t transpose = 0) {}
ALWI void mm_init_short_with_dt(uint32_t in0_cb, uint32_t in1_cb,
                                uint32_t old_in1_cb = 0, uint32_t transpose = 0) {}

// ---- matmul_tiles: tile GEMM accumulate into DST ----
// Reads tile A from CB[in0_cb] at tile offset in0_tile and tile B from
// CB[in1_cb] at tile offset in1_tile.  Accumulates A*B into DST[idst].
// Tiles are 32x32 bfloat16 (2048 bytes each = 1024 uint16_t elements).
// DST stores float32 — acquire zeroes it, then matmul_tiles accumulates.
ALWI void matmul_tiles(uint32_t in0_cb, uint32_t in1_cb,
                       uint32_t in0_tile, uint32_t in1_tile, uint32_t idst) {
    __emule_dst_check(idst, "matmul_tiles");
    __emule_dst_mark_dirty(idst);
    // Standard 32x32 × 32x32 matrix multiply, accumulating into DST[idst].
    // Layout: row-major, 32 rows × 32 cols = 1024 elements per tile.
    constexpr uint32_t DIM = 32;
    float a_rm[DIM * DIM];
    float b_rm[DIM * DIM];
    // Each CB has its own format (bf16 / fp32 / Bfp8_b). Required for
    // reduce-via-matmul where in0 (input data) and in1 (scaler) can differ —
    // e.g. layernorm's first-stage reduce calls matmul_tiles(fp32 × bf16).
    for (uint32_t i = 0; i < DIM * DIM; i++) {
        uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
        a_rm[i] = __emule_read_cb_elem_at(in0_cb, in0_tile, ni);
        b_rm[i] = __emule_read_cb_elem_at(in1_cb, in1_tile, ni);
    }
    // MATH: row-major matmul accumulating into DST.
#ifdef EMULE_MATMUL_USE_AVX2
    for (uint32_t r = 0; r < DIM; r++) {
        for (uint32_t k = 0; k < DIM; k++) {
            __m256 a_vec = _mm256_set1_ps(a_rm[r * DIM + k]);
            for (uint32_t c = 0; c < DIM; c += 8) {
                __m256 b_vec = _mm256_loadu_ps(&b_rm[k * DIM + c]);
                __m256 d_vec = _mm256_loadu_ps(&__emule_dst[idst][r * DIM + c]);
                d_vec = _mm256_fmadd_ps(a_vec, b_vec, d_vec);
                _mm256_storeu_ps(&__emule_dst[idst][r * DIM + c], d_vec);
            }
        }
    }
#else
    for (uint32_t r = 0; r < DIM; r++) {
        for (uint32_t k = 0; k < DIM; k++) {
            float a_val = a_rm[r * DIM + k];
            for (uint32_t c = 0; c < DIM; c++) {
                __emule_dst[idst][r * DIM + c] += a_val * b_rm[k * DIM + c];
            }
        }
    }
#endif
}

// ---- Block matmul stubs (not used by bmm.cpp simple path) ----
ALWI void mm_block_init(uint32_t in0_cb = 0, uint32_t in1_cb = 1,
                        uint32_t out_cb = 16, uint32_t transpose = 0,
                        uint32_t ct_dim = 1, uint32_t rt_dim = 1,
                        uint32_t kt_dim = 1) {}
ALWI void mm_block_init_short(uint32_t in0_cb = 0, uint32_t in1_cb = 1,
                              uint32_t transpose = 0, uint32_t ct_dim = 1,
                              uint32_t rt_dim = 1, uint32_t kt_dim = 1) {}
ALWI void mm_block_init_short_with_dt(uint32_t in0_cb = 0, uint32_t in1_cb = 1,
                                      uint32_t old_in1_cb = 0, uint32_t transpose = 0,
                                      uint32_t ct_dim = 1, uint32_t rt_dim = 1,
                                      uint32_t kt_dim = 1) {}
// matmul_block: one outer-product accumulation over a column of in0 and a row
// of in1, producing a rt_dim × ct_dim block in DST.
//
// Real LLK semantics (tt_metal/hw/inc/api/compute/matmul.h: `matmul_block`):
//   llk_math_matmul<...>(idst, ct_dim, rt_dim)  — math knows nothing of kt_dim.
//   llk_unpack_AB_matmul(in0_cb, in1_cb, in0_idx, in1_idx, ct_dim, rt_dim, kt_dim)
//     — kt_dim is the STRIDE for indexing successive rows of in0 (in0[r] lives
//     at in0_idx + r * kt_dim), NOT a loop count. matmul_block produces
//     exactly ONE outer product per call.
//
// Kernels accumulate the inner dimension by calling matmul_block in a loop,
// stepping in0_idx by 1 and in1_idx by in1_block_w each iteration (see
// `bmm_large_block_zm_fused_bias_activation.cpp` line 320). matmul_tiles
// already accumulates into DST so the outer loop composes correctly.
//
// Tile layout for a single call:
//   in0[in0_tile + r * kt_dim]      — column of in0, row r
//   in1[in1_tile + c]               — row of in1, col c
//   dst[idst   + r * ct_dim + c]    — output, +=
ALWI void matmul_block(uint32_t in0_cb, uint32_t in1_cb,
                       uint32_t in0_tile, uint32_t in1_tile, uint32_t idst,
                       uint32_t transpose = 0, uint32_t ct_dim = 1,
                       uint32_t rt_dim = 1, uint32_t kt_dim = 1) {
    if (rt_dim * ct_dim > 0)
        __emule_dst_check(idst + rt_dim * ct_dim - 1, "matmul_block");
    for (uint32_t r = 0; r < rt_dim; r++) {
        for (uint32_t c = 0; c < ct_dim; c++) {
            matmul_tiles(in0_cb, in1_cb,
                         in0_tile + r * kt_dim,
                         in1_tile + c,
                         idst + r * ct_dim + c);
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
