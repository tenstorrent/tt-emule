// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/hardtanh.h`. Intercepts the upstream
// include path which pulls in `llk_math_eltwise_unary_sfpu_hardtanh.h` (an
// LLK-only header that references SFPU intrinsics).
//
// `param0` and `param1` are uint32 bit-patterns of fp32 low/high bounds,
// matching the host program factory's reinterpret-as-uint32 encoding of the
// hardtanh linear-region endpoints.
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_hardtanh.h
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

}  // namespace ckernel
