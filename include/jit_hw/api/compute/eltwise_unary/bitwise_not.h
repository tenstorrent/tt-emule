// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for bitwise_not SFPU tile op (int32 type-punned).

#include <cstdint>

namespace ckernel {

ALWI void bitwise_not_tile_init() {}

ALWI void bitwise_not_tile(uint32_t idst) {
    __emule_dst_check(idst, "bitwise_not_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, ~v);
    }
}

} // namespace ckernel
