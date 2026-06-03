// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/prelu.h`. `param0` is the uint32
// bit-pattern of an fp32 slope (alpha). Per element: x >= 0 ? x : slope * x.
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_prelu.h
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void prelu_tile_init() {}

ALWI void prelu_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "prelu_tile");
    float slope;
    std::memcpy(&slope, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = (x >= 0.0f) ? x : slope * x;
    }
}

}  // namespace ckernel
