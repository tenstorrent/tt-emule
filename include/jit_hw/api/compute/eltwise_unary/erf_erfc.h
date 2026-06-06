// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stubs for erf / erfc SFPU tile ops.

#include <cmath>
#include <cstdint>

namespace ckernel {

template <bool fast_and_approx = true>
ALWI void erf_tile_init() {}

template <bool fast_and_approx = true>
ALWI void erf_tile(uint32_t idst) {
    __emule_dst_check(idst, "erf_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::erf(__emule_dst[idst][i]);
}

ALWI void erfc_tile_init() {}

ALWI void erfc_tile(uint32_t idst) {
    __emule_dst_check(idst, "erfc_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::erfc(__emule_dst[idst][i]);
}

} // namespace ckernel
