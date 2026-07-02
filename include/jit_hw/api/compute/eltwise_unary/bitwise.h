// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for the SFPU bitwise tile ops (AND/OR/XOR).
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/bitwise.h — a single
// header collapsing the former bitwise_and/bitwise_or/bitwise_xor headers, with
// each op now templated on the integer DataFormat (Int32/UInt32/UInt16). The
// emulated body operates on the int32 DST slot regardless of data_format, so the
// template parameter is accepted and ignored (mirroring where.h/fill.h).

#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

template <DataFormat data_format = DataFormat::Int32>
ALWI void bitwise_and_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "bitwise_and_tile");
    int32_t mask = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, v & mask);
    }
}

template <DataFormat data_format = DataFormat::Int32>
ALWI void bitwise_or_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "bitwise_or_tile");
    int32_t mask = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, v | mask);
    }
}

template <DataFormat data_format = DataFormat::Int32>
ALWI void bitwise_xor_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "bitwise_xor_tile");
    int32_t mask = static_cast<int32_t>(param0);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, v ^ mask);
    }
}

ALWI void bitwise_and_tile_init() {}
ALWI void bitwise_or_tile_init() {}
ALWI void bitwise_xor_tile_init() {}

}  // namespace ckernel
