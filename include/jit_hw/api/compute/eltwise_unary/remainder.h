// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/remainder.h`. Intercepts the
// upstream include path which pulls in `ckernel_sfpu_remainder.h` (an LLK-only
// header that references SFPU intrinsics).
//
// `param0` is the uint32 bit-pattern of the fp32 divisor.
// `param1` is the uint32 bit-pattern of the fp32 reciprocal of the divisor
// (precomputed on-host by silicon; unused here since std::remainder operates
// directly on the divisor with IEEE 754 semantics).
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/eltwise_unary/remainder.h
#include <cmath>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void remainder_tile_init(uint32_t param0, uint32_t param1) {
    // No SFPU state to initialize in the emule path. param0/param1 are
    // accepted to match the upstream signature.
    (void)param0;
    (void)param1;
}

ALWI void remainder_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    __emule_dst_check(idst, "remainder_tile");
    float divisor;
    std::memcpy(&divisor, &param0, sizeof(float));
    (void)param1;  // reciprocal is a silicon-side fast-path hint; unused here.
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_dst[idst][i] = std::remainder(__emule_dst[idst][i], divisor);
    }
}

}  // namespace ckernel
