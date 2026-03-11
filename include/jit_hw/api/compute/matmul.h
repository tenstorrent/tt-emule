#pragma once
// JIT compute stub for matmul operations.
// matmul_tiles performs a 32x32 tile GEMM (bfloat16 inputs, float32 DST accumulate).

#include "jit_hw/api/compute/common.h"

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
    const uint16_t* a_ptr = reinterpret_cast<const uint16_t*>(
        __emule_compute::cb_read_ptr_at(in0_cb, in0_tile));
    const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(
        __emule_compute::cb_read_ptr_at(in1_cb, in1_tile));

    // Standard 32x32 × 32x32 matrix multiply, accumulating into DST[idst].
    // Layout: row-major, 32 rows × 32 cols = 1024 elements per tile.
    constexpr uint32_t DIM = 32;
    for (uint32_t r = 0; r < DIM; r++) {
        for (uint32_t c = 0; c < DIM; c++) {
            float acc = __emule_dst[idst][r * DIM + c];
            for (uint32_t k = 0; k < DIM; k++) {
                acc += __emule_bf16::to_f32(a_ptr[r * DIM + k]) *
                       __emule_bf16::to_f32(b_ptr[k * DIM + c]);
            }
            __emule_dst[idst][r * DIM + c] = acc;
        }
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
