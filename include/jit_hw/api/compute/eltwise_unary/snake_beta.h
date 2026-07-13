// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Ternary SnakeBeta activation: out = x + sin(alpha*x)^2 / beta.
// Real LLK: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_snake_beta.h
// (silicon uses a minimax sin() poly; emule uses std::sin directly.)

#include <cmath>
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void snake_beta_tile_init() {}

template <DataFormat data_format>
ALWI void snake_beta_tile(uint32_t idst_x, uint32_t idst_alpha, uint32_t idst_beta, uint32_t idst_out) {
    __emule_dst_check(idst_x, "snake_beta_tile");
    __emule_dst_check(idst_alpha, "snake_beta_tile");
    __emule_dst_check(idst_beta, "snake_beta_tile");
    __emule_dst_check(idst_out, "snake_beta_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst_x][i];
        float s = std::sin(__emule_compute_ctx().dst[idst_alpha][i] * x);
        __emule_compute_ctx().dst[idst_out][i] = x + (s * s) / __emule_compute_ctx().dst[idst_beta][i];
    }
}

}  // namespace ckernel
