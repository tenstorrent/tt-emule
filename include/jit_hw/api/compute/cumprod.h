// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/cumprod.h`. Mirrors the cumsum API but uses
// multiplicative accumulation. Intercepts the upstream include path which
// pulls in `llk_math_eltwise_unary_sfpu_cumprod.h` (LLK-only, SFPU
// intrinsics).
//
// Real LLK reference:
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_cumprod.h
//
// Semantics: per-thread running product across successive tile-elements.
// `first=true` resets the accumulator to 1.0 before processing the tile.
// For each lane i: acc[i] *= dst[idst][i]; dst[idst][i] = acc[i].
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

// Running product accumulator, one slot per tile element. Persists across
// `cumprod_tile` calls within a thread so multi-tile cumprod chains (NWH
// order, `first=false` for H!=0) accumulate correctly. Initialized to 1.0
// on first use; explicitly reset whenever `first=true`.
static thread_local float __emule_cumprod_acc[__EMULE_TILE_ELEMS] = {};
static thread_local bool __emule_cumprod_acc_initialized = false;

inline void __emule_cumprod_reset_acc() {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_cumprod_acc[i] = 1.0f;
    }
    __emule_cumprod_acc_initialized = true;
}

ALWI void cumprod_tile_init() {
    __emule_cumprod_reset_acc();
}

ALWI void cumprod_tile(uint32_t idst, bool first = true) {
    __emule_dst_check(idst, "cumprod_tile");
    if (first || !__emule_cumprod_acc_initialized) {
        __emule_cumprod_reset_acc();
    }
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_cumprod_acc[i] *= __emule_dst[idst][i];
        __emule_dst[idst][i] = __emule_cumprod_acc[i];
    }
}

}  // namespace ckernel
