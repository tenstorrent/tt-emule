// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Ternary SFPU shim: linear interpolation odst = dst0 + dst2*(dst1 - dst0),
// where dst0=start, dst1=end, dst2=weight.
// Real LLK: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lerp.h

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
        float start = __emule_dst[idst0][i];
        __emule_dst[odst][i] = start + __emule_dst[idst2][i] * (__emule_dst[idst1][i] - start);
    }
}

}  // namespace ckernel
