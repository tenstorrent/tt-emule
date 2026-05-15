// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void where_tile_init() {}

// where_tile: conditional select on DST registers.
// DST[odst][i] = (DST[idst0][i] != 0) ? DST[idst1][i] : DST[idst2][i]
template <DataFormat data_format = DataFormat::Float32>
ALWI void where_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst) {
    __emule_dst_check(idst0, "where_tile");
    __emule_dst_check(idst1, "where_tile");
    __emule_dst_check(idst2, "where_tile");
    __emule_dst_check(odst, "where_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_dst[odst][i] = (__emule_dst[idst0][i] != 0.0f)
                                ? __emule_dst[idst1][i]
                                : __emule_dst[idst2][i];
    }
}

} // namespace ckernel
