// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT compute stub for matmul operations.
// matmul_tiles performs a 32x32 tile GEMM (bfloat16 inputs, float32 DST accumulate).
// Uses AVX2/FMA intrinsics when available for ~4-8x speedup over scalar.

#include "jit_hw/api/compute/bfp8.h"
#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/nfaces.h"
#include "jit_hw/internal/llk_state.h"
#include "jit_hw/llk/llk_reduce_primitives.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define EMULE_MATMUL_USE_AVX2 1
#endif

namespace ckernel {

// ---- Init stubs (hardware pipeline configuration) ----
// `transpose=1` is honored: stored in `__llk_matmul_transpose` and applied
// to the IN1 tile in matmul_tiles below — see internal/llk_state.h.
ALWI void mm_init(uint32_t in0_cb = 0, uint32_t in1_cb = 1, uint32_t out_cb = 16,
                  uint32_t transpose = 0) {
    __llk_matmul_transpose = (transpose != 0);
}
ALWI void mm_init_short(uint32_t in0_cb = 0, uint32_t in1_cb = 1,
                        uint32_t transpose = 0) {
    __llk_matmul_transpose = (transpose != 0);
}
ALWI void mm_init_short_with_dt(uint32_t in0_cb, uint32_t in1_cb,
                                uint32_t old_in1_cb = 0, uint32_t transpose = 0) {
    __llk_matmul_transpose = (transpose != 0);
}

// ---- matmul_tiles: tile GEMM accumulate into DST ----
// Reads tile A from CB[in0_cb] at tile offset in0_tile and tile B from
// CB[in1_cb] at tile offset in1_tile.  Accumulates A*B into DST[idst].
// DST stores float32 — acquire zeroes it, then matmul_tiles accumulates.
// Per-operand format dispatch: fp32 / bf16 / bfp8_b decoded independently for
// A and B (silicon's unpacker reconfig is per-operand; the prior single-format
// branch on in0_cb mis-routed bfp8 in1 into the bf16 reader, walking past-end
// memory and producing ~50% NaN).
ALWI void matmul_tiles(uint32_t in0_cb, uint32_t in1_cb,
                       uint32_t in0_tile, uint32_t in1_tile, uint32_t idst) {
    __emule_dst_check(idst, "matmul_tiles");
    __emule_dst_mark_dirty(idst);
    // Standard 32x32 × 32x32 matrix multiply, accumulating into DST[idst].
    // Layout: row-major, 32 rows × 32 cols = 1024 elements per tile.
    constexpr uint32_t DIM = 32;
    float a_rm[DIM * DIM];
    float b_rm[DIM * DIM];
    auto decode_to_rm = [](uint32_t cb, uint32_t tile_idx, float* rm_out) {
        const uint8_t* base = __emule_compute::cb_read_ptr_at(cb, tile_idx);
        if (__emule_compute::cb_is_32bit_format(cb)) {
            const float* p = reinterpret_cast<const float*>(base);
            for (uint32_t i = 0; i < DIM * DIM; i++) {
                uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
                rm_out[i] = p[ni];
            }
        } else if (__emule_compute::cb_is_bfp8_b_format(cb)) {
            for (uint32_t i = 0; i < DIM * DIM; i++) {
                uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
                rm_out[i] = __emule_bfp8::to_f32(base, ni);
            }
        } else {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(base);
            for (uint32_t i = 0; i < DIM * DIM; i++) {
                uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
                rm_out[i] = __emule_bf16::to_f32(p[ni]);
            }
        }
    };
    decode_to_rm(in0_cb, in0_tile, a_rm);
    decode_to_rm(in1_cb, in1_tile, b_rm);
    // Apply IN1 transpose if mm_init(... transpose=1) was set — silicon does
    // this in the unpacker (THCON_SEC0_REG2_Haloize_mode_RMW); emule transposes
    // the decoded row-major view in-place before the FMA loop.
    if (__llk_matmul_transpose) {
        for (uint32_t r = 0; r < DIM; r++) {
            for (uint32_t c = r + 1; c < DIM; c++) {
                std::swap(b_rm[r * DIM + c], b_rm[c * DIM + r]);
            }
        }
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
                        uint32_t kt_dim = 1) {
    __llk_matmul_transpose = (transpose != 0);
}
ALWI void mm_block_init_short(uint32_t in0_cb = 0, uint32_t in1_cb = 1,
                              uint32_t transpose = 0, uint32_t ct_dim = 1,
                              uint32_t rt_dim = 1, uint32_t kt_dim = 1) {
    __llk_matmul_transpose = (transpose != 0);
}
ALWI void mm_block_init_short_with_dt(uint32_t in0_cb = 0, uint32_t in1_cb = 1,
                                      uint32_t old_in1_cb = 0, uint32_t transpose = 0,
                                      uint32_t ct_dim = 1, uint32_t rt_dim = 1,
                                      uint32_t kt_dim = 1) {
    __llk_matmul_transpose = (transpose != 0);
}
// matmul_block: compute rt_dim × ct_dim block of output tiles.
// For each output tile (r, c): DST[idst + r*ct_dim + c] += A[in0_tile + r*kt_dim] * B[in1_tile + c]
// The runtime `transpose` arg here overrides any value previously set by
// mm_block_init for this block call.
ALWI void matmul_block(uint32_t in0_cb, uint32_t in1_cb,
                       uint32_t in0_tile, uint32_t in1_tile, uint32_t idst,
                       uint32_t transpose = 0, uint32_t ct_dim = 1,
                       uint32_t rt_dim = 1, uint32_t kt_dim = 1) {
    if (rt_dim * ct_dim > 0)
        __emule_dst_check(idst + rt_dim * ct_dim - 1, "matmul_block");
    __llk_matmul_transpose = (transpose != 0);
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
