// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for upstream tt_metal/hw/inc/api/compute/eltwise_unary/bitwise_xor.h.

#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void bitwise_xor_tile_init() {}

ALWI void bitwise_xor_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "bitwise_xor_tile");
    int32_t mask = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, v ^ mask);
    }
}

} // namespace ckernel
