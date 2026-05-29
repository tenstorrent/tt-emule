// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
#include <algorithm>

namespace ckernel {

ALWI void binary_max_tile_init() {}
ALWI void binary_min_tile_init() {}
ALWI void binary_max_int32_tile_init() {}
ALWI void binary_min_int32_tile_init() {}
ALWI void binary_max_uint32_tile_init() {}
ALWI void binary_min_uint32_tile_init() {}

ALWI void binary_max_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[odst][i] = std::max(__emule_dst[idst0][i], __emule_dst[idst1][i]);
}
// 4-arg overload: trailing vector_mode (real signature accepts an int
// VectorMode flag for selective row/column/face application). Emule treats
// the full tile uniformly — mode flag is ignored.
ALWI void binary_max_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, int /*vector_mode*/) {
    binary_max_tile(idst0, idst1, odst);
}

ALWI void binary_min_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[odst][i] = std::min(__emule_dst[idst0][i], __emule_dst[idst1][i]);
}
ALWI void binary_min_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, int /*vector_mode*/) {
    binary_min_tile(idst0, idst1, odst);
}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void binary_max_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, a > b ? a : b);
    }
}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void binary_min_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, a < b ? a : b);
    }
}

template<DataFormat Fmt = DataFormat::UInt32>
ALWI void binary_max_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        uint32_t a, b;
        std::memcpy(&a, &__emule_dst[idst0][i], 4);
        std::memcpy(&b, &__emule_dst[idst1][i], 4);
        uint32_t r = a > b ? a : b;
        std::memcpy(&__emule_dst[odst][i], &r, 4);
    }
}

template<DataFormat Fmt = DataFormat::UInt32>
ALWI void binary_min_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        uint32_t a, b;
        std::memcpy(&a, &__emule_dst[idst0][i], 4);
        std::memcpy(&b, &__emule_dst[idst1][i], 4);
        uint32_t r = a < b ? a : b;
        std::memcpy(&__emule_dst[odst][i], &r, 4);
    }
}

} // namespace ckernel
