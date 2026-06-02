// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/lerp.h`. Per element:
// out = input + weight * (end - input), with idst0=input, idst1=end, idst2=weight.

#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void lerp_tile_init() {}

template <DataFormat data_format>
ALWI void lerp_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst) {
    __emule_dst_check(idst0, "lerp_tile");
    __emule_dst_check(idst1, "lerp_tile");
    __emule_dst_check(idst2, "lerp_tile");
    __emule_dst_check(odst, "lerp_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        const float input = __emule_dst[idst0][i];
        const float end = __emule_dst[idst1][i];
        const float weight = __emule_dst[idst2][i];
        __emule_dst[odst][i] = input + weight * (end - input);
    }
}

}  // namespace ckernel
