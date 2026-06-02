// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/mask.h`. Per element: mask==0 zeros the data
// element (mask_tile) or sets it to +inf (mask_posinf_tile).
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_mask.h

#include <cstdint>
#include <limits>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void mask_tile_init() {}

ALWI void mask_tile(uint32_t idst_data, uint32_t idst_mask, DataFormat data_format = DataFormat::Float16_b) {
    __emule_dst_check(idst_data, "mask_tile");
    __emule_dst_check(idst_mask, "mask_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        if (__emule_dst[idst_mask][i] == 0.0f) {
            __emule_dst[idst_data][i] = 0.0f;
        }
    }
}

ALWI void mask_posinf_tile(uint32_t idst_data, uint32_t idst_mask) {
    __emule_dst_check(idst_data, "mask_posinf_tile");
    __emule_dst_check(idst_mask, "mask_posinf_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        if (__emule_dst[idst_mask][i] == 0.0f) {
            __emule_dst[idst_data][i] = std::numeric_limits<float>::infinity();
        }
    }
}

}  // namespace ckernel
