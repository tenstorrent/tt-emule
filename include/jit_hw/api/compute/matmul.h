#pragma once
// JIT compute stub for matmul operations.
// matmul_tiles performs a 32x32 tile GEMM (bfloat16 inputs, float32 DST accumulate).
// Uses AVX2/FMA intrinsics when available for ~4-8x speedup over scalar.

#include "jit_hw/api/compute/common.h"

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
    // Standard 32x32 × 32x32 matrix multiply, accumulating into DST[idst].
    // Layout: row-major, 32 rows × 32 cols = 1024 elements per tile.
    constexpr uint32_t DIM = 32;
    if (__emule_compute::cb_is_32bit_format(in0_cb)) {
        // Float32 path: read tiles as float* directly.
        const float* a_ptr = reinterpret_cast<const float*>(
            __emule_compute::cb_read_ptr_at(in0_cb, in0_tile));
        const float* b_ptr = reinterpret_cast<const float*>(
            __emule_compute::cb_read_ptr_at(in1_cb, in1_tile));
#ifdef EMULE_MATMUL_USE_AVX2
        // AVX2+FMA path: broadcast a_val, FMA over 8 floats at a time (32/8 = 4 iters).
        for (uint32_t r = 0; r < DIM; r++) {
            for (uint32_t k = 0; k < DIM; k++) {
                __m256 a_vec = _mm256_set1_ps(a_ptr[r * DIM + k]);
                for (uint32_t c = 0; c < DIM; c += 8) {
                    __m256 b_vec = _mm256_loadu_ps(&b_ptr[k * DIM + c]);
                    __m256 d_vec = _mm256_loadu_ps(&__emule_dst[idst][r * DIM + c]);
                    d_vec = _mm256_fmadd_ps(a_vec, b_vec, d_vec);
                    _mm256_storeu_ps(&__emule_dst[idst][r * DIM + c], d_vec);
                }
            }
        }
#else
        // Scalar fallback: loop order r,k,c for sequential B access (cache-friendly).
        for (uint32_t r = 0; r < DIM; r++) {
            for (uint32_t k = 0; k < DIM; k++) {
                float a_val = a_ptr[r * DIM + k];
                for (uint32_t c = 0; c < DIM; c++) {
                    __emule_dst[idst][r * DIM + c] += a_val * b_ptr[k * DIM + c];
                }
            }
        }
#endif
    } else {
        // bfloat16 path: pre-convert B tile to f32 for cache-friendly access.
        const uint16_t* a_ptr = reinterpret_cast<const uint16_t*>(
            __emule_compute::cb_read_ptr_at(in0_cb, in0_tile));
        const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(
            __emule_compute::cb_read_ptr_at(in1_cb, in1_tile));
        float b_f32[DIM * DIM];
#ifdef EMULE_MATMUL_USE_AVX2
        // AVX2 bulk bf16→f32 conversion for B tile.
        for (uint32_t i = 0; i < DIM * DIM; i += 8) {
            // Convert 8 bf16 values to f32 via shift-left-16.
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b_ptr[i]));
            __m256i wide = _mm256_cvtepu16_epi32(raw);
            __m256i shifted = _mm256_slli_epi32(wide, 16);
            _mm256_storeu_ps(&b_f32[i], _mm256_castsi256_ps(shifted));
        }
        // AVX2+FMA matmul with bf16 A (convert per-element) and pre-converted B.
        for (uint32_t r = 0; r < DIM; r++) {
            for (uint32_t k = 0; k < DIM; k++) {
                float a_val = __emule_bf16::to_f32(a_ptr[r * DIM + k]);
                __m256 a_vec = _mm256_set1_ps(a_val);
                for (uint32_t c = 0; c < DIM; c += 8) {
                    __m256 b_vec = _mm256_loadu_ps(&b_f32[k * DIM + c]);
                    __m256 d_vec = _mm256_loadu_ps(&__emule_dst[idst][r * DIM + c]);
                    d_vec = _mm256_fmadd_ps(a_vec, b_vec, d_vec);
                    _mm256_storeu_ps(&__emule_dst[idst][r * DIM + c], d_vec);
                }
            }
        }
#else
        // Scalar bf16→f32 conversion for B tile.
        for (uint32_t i = 0; i < DIM * DIM; i++) {
            b_f32[i] = __emule_bf16::to_f32(b_ptr[i]);
        }
        // Scalar fallback: loop order r,k,c for sequential B access.
        for (uint32_t r = 0; r < DIM; r++) {
            for (uint32_t k = 0; k < DIM; k++) {
                float a_val = __emule_bf16::to_f32(a_ptr[r * DIM + k]);
                for (uint32_t c = 0; c < DIM; c++) {
                    __emule_dst[idst][r * DIM + c] += a_val * b_f32[k * DIM + c];
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
