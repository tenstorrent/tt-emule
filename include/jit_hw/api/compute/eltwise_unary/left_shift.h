// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for left_shift SFPU tile op (upstream:
// tt_metal/hw/inc/api/compute/eltwise_unary/left_shift.h).

#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void left_shift_tile_init() {}

ALWI void left_shift_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "left_shift_tile");
    uint32_t shift = param0 & 0x1F;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        uint32_t u = static_cast<uint32_t>(v) << shift;
        __emule_dst_store_i32(idst, i, static_cast<int32_t>(u));
    }
}

}  // namespace ckernel
