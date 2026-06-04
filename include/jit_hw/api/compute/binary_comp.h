// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// INT32 comparison SFPU stubs.
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void lt_int32_tile_init() {}
ALWI void gt_int32_tile_init() {}
ALWI void ge_int32_tile_init() {}
ALWI void le_int32_tile_init() {}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void lt_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, a < b ? 1 : 0);
    }
}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void gt_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, a > b ? 1 : 0);
    }
}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void ge_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, a >= b ? 1 : 0);
    }
}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void le_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, a <= b ? 1 : 0);
    }
}

// ---- Upstream-named int comparisons → 1/0 (binary_comp.h). DataFormat selects
// signed (Int32) vs unsigned (UInt32/UInt16) compare on the DST int bits.
// data_format == UInt16 ? (uint16_t)a < (uint16_t)b : etc.
// Only lt/gt/le/ge are provided: the binary_ng host emits `*_int_tile` only for the ordered
// comparisons (binary_ng_utils.cpp). Integer eq/ne don't route here — equality is bit-exact
// so they take the format-agnostic path (verified by test_relational's edge_case with
// INT_MAX). Init forms are templated on DataFormat to match the host's emitted call
// (binary_ng_utils.cpp: `lt_int_tile_init<DataFormat::Int32>();`).
template <DataFormat data_format> ALWI void lt_int_tile_init() {}
template <DataFormat data_format> ALWI void gt_int_tile_init() {}
template <DataFormat data_format> ALWI void le_int_tile_init() {}
template <DataFormat data_format> ALWI void ge_int_tile_init() {}

template <DataFormat data_format>
ALWI void lt_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i), b = __emule_dst_load_i32(idst1, i);
        bool r;
        if constexpr (data_format == DataFormat::UInt32) r = (uint32_t)a < (uint32_t)b;
        else if constexpr (data_format == DataFormat::UInt16) r = (uint16_t)a < (uint16_t)b;
        else r = a < b;
        __emule_dst_store_i32(odst, i, r ? 1 : 0);
    }
}
template <DataFormat data_format>
ALWI void gt_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i), b = __emule_dst_load_i32(idst1, i);
        bool r;
        if constexpr (data_format == DataFormat::UInt32) r = (uint32_t)a > (uint32_t)b;
        else if constexpr (data_format == DataFormat::UInt16) r = (uint16_t)a > (uint16_t)b;
        else r = a > b;
        __emule_dst_store_i32(odst, i, r ? 1 : 0);
    }
}
template <DataFormat data_format>
ALWI void le_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i), b = __emule_dst_load_i32(idst1, i);
        bool r;
        if constexpr (data_format == DataFormat::UInt32) r = (uint32_t)a <= (uint32_t)b;
        else if constexpr (data_format == DataFormat::UInt16) r = (uint16_t)a <= (uint16_t)b;
        else r = a <= b;
        __emule_dst_store_i32(odst, i, r ? 1 : 0);
    }
}
template <DataFormat data_format>
ALWI void ge_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i), b = __emule_dst_load_i32(idst1, i);
        bool r;
        if constexpr (data_format == DataFormat::UInt32) r = (uint32_t)a >= (uint32_t)b;
        else if constexpr (data_format == DataFormat::UInt16) r = (uint16_t)a >= (uint16_t)b;
        else r = a >= b;
        __emule_dst_store_i32(odst, i, r ? 1 : 0);
    }
}

} // namespace ckernel
