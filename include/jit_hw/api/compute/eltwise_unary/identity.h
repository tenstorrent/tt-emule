// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/identity.h`. y(x) = x; the
// uint32 variant operates on DST reinterpreted as int32.
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_identity.h
#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void identity_tile_init() {}

ALWI void identity_tile(uint32_t idst) {
    __emule_dst_check(idst, "identity_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_dst[idst][i] = __emule_dst[idst][i];
    }
}

ALWI void identity_tile_uint32(uint32_t idst) {
    __emule_dst_check(idst, "identity_tile_uint32");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, v);
    }
}

}  // namespace ckernel
