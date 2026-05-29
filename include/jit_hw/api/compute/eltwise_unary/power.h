// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator shim for power_iterative_tile SFPU op (integer-power).
// SDPA / moreh kernels use this for raising tile elements to a (compile-time
// or runtime) integer power. Implemented as repeated multiply.

#include <cstdint>
#include "jit_hw/api/compute/common.h"

namespace ckernel {

inline void power_iterative_tile_init() {}

inline void power_iterative_tile(uint32_t idst, uint32_t pow) {
    __emule_dst_check(idst, "power_iterative_tile");
    if (pow == 0) {
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
            __emule_dst[idst][i] = 1.0f;
        return;
    }
    float base[__EMULE_TILE_ELEMS];
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) base[i] = __emule_dst[idst][i];
    for (uint32_t p = 1; p < pow; ++p) {
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
            __emule_dst[idst][i] *= base[i];
    }
}

}  // namespace ckernel
