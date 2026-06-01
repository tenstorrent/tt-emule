// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/rsub.h`. `param0` is the uint32
// bit-pattern of an fp32 scalar; output = scalar - x.
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_rsub.h
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void rsub_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "rsub_tile");
    float scalar;
    std::memcpy(&scalar, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_dst[idst][i] = scalar - __emule_dst[idst][i];
    }
}

ALWI void rsub_tile_init() {}

}  // namespace ckernel
