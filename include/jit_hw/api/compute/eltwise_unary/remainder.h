// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/remainder.h`.
// `param0` is the uint32 bit-pattern of the fp32 divisor; `param1` is the
// reciprocal (silicon fast-path hint, unused by std::remainder).
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
        // torch.remainder: a - b*floor(a/b) (sign follows divisor). NOT
        // std::remainder, which is IEEE round-to-nearest (sign follows dividend).
        const float a = __emule_dst[idst][i];
        __emule_dst[idst][i] = a - divisor * std::floor(a / divisor);
    }
}

}  // namespace ckernel
