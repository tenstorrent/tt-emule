// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// matmul_custom (no-MOP matmul) compute shim for emulated mode.
//
// The real api/compute/experimental/matmul_custom.h pulls in the experimental LLK
// matmul-custom headers (raw-TTI Tensix matmul engine) which emule does not compile.
// The "no-MOP" path is a HW execution strategy (direct replay buffer instead of a MOP)
// with NO numerical effect — same C=A*B as the standard matmul_block. So delegate to
// emule's matmul_block / mm_block_init_short; the MM_THROTTLE / no-MOP machinery is
// HW-only and irrelevant to the scalar model.
//
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/experimental/llk_math_matmul_custom.h

#include "jit_hw/api/compute/matmul.h"

#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif

namespace ckernel {

ALWI void mm_no_mop_init_short(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const bool transpose = false,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1) {
    mm_block_init_short(in0_cb_id, in1_cb_id, (uint32_t)transpose, ct_dim, rt_dim, kt_dim);
}

ALWI void matmul_block_no_mop(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const bool transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim) {
    matmul_block(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, idst, (uint32_t)transpose, ct_dim, rt_dim, kt_dim);
}

ALWI void mm_no_mop_reinit_short(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const bool transpose = false,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1) {
    mm_block_init_short(in0_cb_id, in1_cb_id, (uint32_t)transpose, ct_dim, rt_dim, kt_dim);
}

}  // namespace ckernel
