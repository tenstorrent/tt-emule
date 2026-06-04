// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator implementation of clamp SFPU tile op.
// Mirrors silicon TTKernel_ClampScalarTileOp / TTKernel_ClampScalarTileInt32Op
// in tt-mlir/include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td.

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void clamp_tile_init() {}

// Float variant: min_param / max_param are uint32_t bit-casts of float bounds.
ALWI void clamp_tile(uint32_t idst, uint32_t min_param, uint32_t max_param) {
    __emule_dst_check(idst, "clamp_tile");
    float min_val, max_val;
    std::memcpy(&min_val, &min_param, sizeof(float));
    std::memcpy(&max_val, &max_param, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float v = __emule_dst[idst][i];
        __emule_dst[idst][i] = std::min(std::max(v, min_val), max_val);
    }
}

// INT32 variant: min_param / max_param are int32_t reinterpreted as uint32_t.
ALWI void clamp_tile_int32(uint32_t idst, uint32_t min_param, uint32_t max_param) {
    __emule_dst_check(idst, "clamp_tile_int32");
    int32_t min_val = static_cast<int32_t>(min_param);
    int32_t max_val = static_cast<int32_t>(max_param);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, std::min(std::max(v, min_val), max_val));
    }
}

} // namespace ckernel
