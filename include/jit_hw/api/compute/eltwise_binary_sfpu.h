// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// SFPU binary ops (float DST-to-DST): add, sub, mul, rsub, div.
#include "jit_hw/api/compute/common.h"

namespace ckernel {

// ---- Init stubs (no-ops) ----
ALWI void add_binary_tile_init() {}
ALWI void sub_binary_tile_init() {}
ALWI void mul_binary_tile_init() {}
ALWI void rsub_binary_tile_init() {}
ALWI void div_binary_tile_init() {}
ALWI void power_binary_tile_init() {}

// ---- DST-to-DST float binary ops ----

ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[odst][i] = __emule_dst[idst0][i] + __emule_dst[idst1][i];
}

ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[odst][i] = __emule_dst[idst0][i] - __emule_dst[idst1][i];
}

ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[odst][i] = __emule_dst[idst0][i] * __emule_dst[idst1][i];
}

ALWI void rsub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[odst][i] = __emule_dst[idst1][i] - __emule_dst[idst0][i];
}

ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[odst][i] = __emule_dst[idst1][i] != 0.f
            ? __emule_dst[idst0][i] / __emule_dst[idst1][i] : 0.f;
}

ALWI void power_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[odst][i] = std::pow(__emule_dst[idst0][i], __emule_dst[idst1][i]);
}

} // namespace ckernel
