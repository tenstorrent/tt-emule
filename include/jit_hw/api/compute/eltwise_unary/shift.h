// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for the SFPU shift tile ops (left_shift/right_shift).
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/shift.h — a single header
// collapsing the former left_shift/right_shift headers, with each op now
// templated on the integer DataFormat (Int32/UInt32/UInt16).
//
// Canonical LLK: ckernel_sfpu_unary_shift.h. left_shift is a logical shift;
// right_shift is arithmetic (sign-replicating). The emulated body operates on
// the int32 DST slot regardless of data_format, so the template parameter is
// accepted and ignored (mirroring where.h/fill.h).

#include <cstdint>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

template <DataFormat data_format = DataFormat::Int32>
ALWI void left_shift_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "left_shift_tile");
    const bool out_of_range = param0 >= 32u;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        uint32_t u = out_of_range ? 0u : (static_cast<uint32_t>(v) << param0);
        __emule_dst_store_i32(idst, i, static_cast<int32_t>(u));
    }
}

template <DataFormat data_format = DataFormat::Int32>
ALWI void right_shift_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "right_shift_tile");
    const bool out_of_range = param0 >= 32u;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        // Arithmetic right shift; a shift >= 32 saturates to the sign
        // (non-negative -> 0, negative -> -1), matching ckernel_sfpu_unary_shift.h.
        int32_t r = out_of_range ? (v < 0 ? -1 : 0) : (v >> param0);
        __emule_dst_store_i32(idst, i, r);
    }
}

ALWI void left_shift_tile_init() {}
ALWI void right_shift_tile_init() {}

}  // namespace ckernel
