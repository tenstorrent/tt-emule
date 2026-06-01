// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/threshold.h`. Intercepts the
// upstream include path which pulls in `llk_math_eltwise_unary_sfpu_threshold.h`
// (an LLK-only header that references SFPU intrinsics).
//
// `param0` is the uint32 bit-pattern of an fp32 threshold value.
// `param1` is the uint32 bit-pattern of an fp32 replacement value.
// Per element: output = x > threshold ? x : value.
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/eltwise_unary/threshold.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_threshold.h
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void threshold_tile_init() {}

ALWI void threshold_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    __emule_dst_check(idst, "threshold_tile");
    float threshold;
    float value;
    std::memcpy(&threshold, &param0, sizeof(float));
    std::memcpy(&value, &param1, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = (x > threshold) ? x : value;
    }
}

}  // namespace ckernel
