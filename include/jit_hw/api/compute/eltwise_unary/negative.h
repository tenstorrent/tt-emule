// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stub for negation tile operations.

#include <cstdint>

namespace ckernel {

ALWI void negative_tile_init() {}

ALWI void negative_tile(uint32_t idst) {
    __emule_dst_check(idst, "negative_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = -__emule_dst[idst][i];
}

ALWI void negative_tile_int32(uint32_t idst) {
    __emule_dst_check(idst, "negative_tile_int32");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, -v);
    }
}

} // namespace ckernel
