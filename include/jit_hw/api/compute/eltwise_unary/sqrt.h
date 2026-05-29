// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for sqrt SFPU tile op.

#include <cmath>
#include <cstdint>

namespace ckernel {

template <bool FAST_APPROX = false>
ALWI void sqrt_tile_init() {}

template <bool FAST_APPROX = false>
ALWI void sqrt_tile(uint32_t idst) {
    __emule_dst_check(idst, "sqrt_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::sqrt(__emule_dst[idst][i]);
}

} // namespace ckernel
