// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/cumprod.h`. Per-thread running product across
// tile-elements; `first=true` resets the accumulator to 1.0.
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_cumprod.h
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

// Persists across cumprod_tile calls so multi-tile chains (NWH order,
// first=false on H!=0) accumulate.

inline void __emule_cumprod_reset_acc() {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_compute_ctx().cumprod_acc[i] = 1.0f;
    }
    __emule_compute_ctx().cumprod_acc_initialized = true;
}

ALWI void cumprod_tile_init() {
    __emule_cumprod_reset_acc();
}

ALWI void cumprod_tile(uint32_t idst, bool first = true) {
    __emule_dst_check(idst, "cumprod_tile");
    if (first || !__emule_compute_ctx().cumprod_acc_initialized) {
        __emule_cumprod_reset_acc();
    }
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_compute_ctx().cumprod_acc[i] *= __emule_compute_ctx().dst[idst][i];
        __emule_compute_ctx().dst[idst][i] = __emule_compute_ctx().cumprod_acc[i];
    }
}

}  // namespace ckernel
