// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/hardtanh.h`.
// `param0`/`param1` are uint32 bit-patterns of fp32 low/high bounds.
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_hardtanh.h
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void hardtanh_tile_init() {}

ALWI void hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    __emule_dst_check(idst, "hardtanh_tile");
    float low;
    float high;
    std::memcpy(&low, &param0, sizeof(float));
    std::memcpy(&high, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        if (x < low) {
            x = low;
        } else if (x > high) {
            x = high;
        }
        __emule_dst[idst][i] = x;
    }
}

// Pack-thread variant: upstream dispatches the same SFPU LLK via PACK() for
// matmul fused-activation overlap (#79). emule has no thread split — forward.
ALWI void hardtanh_tile_init_pack() { hardtanh_tile_init(); }
ALWI void hardtanh_tile_pack(uint32_t idst, uint32_t param0, uint32_t param1) {
    hardtanh_tile(idst, param0, param1);
}

}  // namespace ckernel
