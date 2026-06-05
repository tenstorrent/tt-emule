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
// Track in1 transpose for matmul_tiles. Real HW configures the UNPACK engine
// in mm_init; on emule we just remember the flag.
inline thread_local uint32_t __emule_mm_transpose_in1 = 0;
ALWI void mm_init(uint32_t in0_cb = 0, uint32_t in1_cb = 1, uint32_t out_cb = 16,
                  uint32_t transpose = 0) { __emule_mm_transpose_in1 = transpose; }
ALWI void mm_init_short(uint32_t in0_cb = 0, uint32_t in1_cb = 1,
                        uint32_t transpose = 0) { __emule_mm_transpose_in1 = transpose; }
ALWI void mm_init_short_with_dt(uint32_t in0_cb, uint32_t in1_cb,
                                uint32_t old_in1_cb = 0, uint32_t transpose = 0) {
    __emule_mm_transpose_in1 = transpose;
}

// ---- matmul_tiles: tile GEMM accumulate into DST ----
// Reads tile A from CB[in0_cb] at tile offset in0_tile and tile B from
// CB[in1_cb] at tile offset in1_tile.  Accumulates A*B into DST[idst].
// Tiles are 32x32 bfloat16 (2048 bytes each = 1024 uint16_t elements).
// DST stores float32 — acquire zeroes it, then matmul_tiles accumulates.
ALWI void matmul_tiles(uint32_t in0_cb, uint32_t in1_cb,
                       uint32_t in0_tile, uint32_t in1_tile, uint32_t idst) {
    __emule_dst_check(idst, "matmul_tiles");
    __emule_dst_mark_dirty(idst);
    // Tile-aware GEMM accumulate into DST[idst].
    //
    // Silicon's custom_mm_block supports in0 tile shapes [{1,2,4,8,16,32}, 32]
    // and in1 tile shape [32, 32]. The in0 page may be smaller than 1024
    // elements when M < 32 (e.g. the MatmulFusedAct kernel uses 1×32 act tiles).
    //
    // We derive `rows_A` from the in0 CB page size; rows_B always = 32; cols
    // always = 32. DST is always full 32×32; rows beyond rows_A stay zero
    // (the test reads only the meaningful rows).
    constexpr uint32_t DIM = 32;
    const bool is_32bit = __emule_compute::cb_is_32bit_format(in0_cb);
    const uint32_t elem_bytes_a = is_32bit ? 4u : 2u;
    const uint32_t page_a = __emule_compute::cb_page_size(in0_cb);
    const uint32_t rows_a = __emule_nfaces::tile_rows_from_pagesize(page_a, elem_bytes_a);

    float a_rm[DIM * DIM] = {};  // rows_a × 32, rest zero (unused)
    float b_rm[DIM * DIM];

    if (is_32bit) {
        const float* a_ptr = reinterpret_cast<const float*>(
            __emule_compute::cb_read_ptr_at(in0_cb, in0_tile));
        const float* b_ptr = reinterpret_cast<const float*>(
            __emule_compute::cb_read_ptr_at(in1_cb, in1_tile));
        for (uint32_t i = 0; i < rows_a * DIM; i++) {
            a_rm[i] = a_ptr[__emule_nfaces::tile_rm_to_nfaces(i, rows_a)];
        }
        for (uint32_t i = 0; i < DIM * DIM; i++) {
            b_rm[i] = b_ptr[__emule_nfaces::rowmajor_to_nfaces[i]];
        }
    } else {
        const uint16_t* a_ptr = reinterpret_cast<const uint16_t*>(
            __emule_compute::cb_read_ptr_at(in0_cb, in0_tile));
        const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(
            __emule_compute::cb_read_ptr_at(in1_cb, in1_tile));
        for (uint32_t i = 0; i < rows_a * DIM; i++) {
            a_rm[i] = __emule_bf16::to_f32(
                a_ptr[__emule_nfaces::tile_rm_to_nfaces(i, rows_a)]);
        }
        for (uint32_t i = 0; i < DIM * DIM; i++) {
            b_rm[i] = __emule_bf16::to_f32(b_ptr[__emule_nfaces::rowmajor_to_nfaces[i]]);
        }
    }
    // MATH: row-major matmul accumulating into DST.
    // If in1 was configured as transposed in mm_init, treat B as B^T:
    //   DST[r,c] += sum_k A[r,k] * B[c,k].
    if (__emule_mm_transpose_in1) {
        for (uint32_t r = 0; r < rows_a; r++) {
            for (uint32_t c = 0; c < DIM; c++) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < DIM; k++) {
                    acc += a_rm[r * DIM + k] * b_rm[c * DIM + k];
                }
                __emule_dst[idst][r * DIM + c] += acc;
            }
        }
    } else {
#ifdef EMULE_MATMUL_USE_AVX2
        for (uint32_t r = 0; r < rows_a; r++) {
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
        for (uint32_t r = 0; r < rows_a; r++) {
            for (uint32_t k = 0; k < DIM; k++) {
                float a_val = a_rm[r * DIM + k];
                for (uint32_t c = 0; c < DIM; c++) {
                    __emule_dst[idst][r * DIM + c] += a_val * b_rm[k * DIM + c];
                }
            }
        }
#endif
    }
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
// matmul_block: compute rt_dim × ct_dim block of output tiles.
// For each output tile (r, c): DST[idst + r*ct_dim + c] += A[in0_tile + r*kt_dim] * B[in1_tile + c]
ALWI void matmul_block(uint32_t in0_cb, uint32_t in1_cb,
                       uint32_t in0_tile, uint32_t in1_tile, uint32_t idst,
                       uint32_t transpose = 0, uint32_t ct_dim = 1,
                       uint32_t rt_dim = 1, uint32_t kt_dim = 1) {
    if (rt_dim * ct_dim > 0)
        __emule_dst_check(idst + rt_dim * ct_dim - 1, "matmul_block");
    uint32_t dst = idst;
    for (uint32_t r = 0; r < rt_dim; r++) {
        for (uint32_t c = 0; c < ct_dim; c++) {
            matmul_tiles(in0_cb, in1_cb,
                         in0_tile + r * kt_dim,
                         in1_tile + c,
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
