// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stubs for ReLU SFPU tile ops.

#include <algorithm>
#include <cstdint>

namespace ckernel {

ALWI void relu_tile_init() {}

ALWI void relu_tile(uint32_t idst) {
    __emule_dst_check(idst, "relu_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::max(0.0f, __emule_dst[idst][i]);
}

ALWI void relu_tile_int32(uint32_t idst) {
    __emule_dst_check(idst, "relu_tile_int32");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, std::max<int32_t>(0, v));
    }
}

} // namespace ckernel
