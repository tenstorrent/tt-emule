// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void fmod_binary_tile_init() {}
ALWI void fmod_int32_tile_init() {}

ALWI void fmod_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[odst][i] = std::fmod(__emule_dst[idst0][i], __emule_dst[idst1][i]);
}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void fmod_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, b != 0 ? a % b : 0);
    }
}

} // namespace ckernel
