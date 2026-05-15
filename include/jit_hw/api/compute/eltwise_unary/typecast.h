// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for typecast tile operations.
// Converts between data formats within DST (float32 storage with type-pun for int32).

#include <cstdint>
#include <cstring>
#include <cmath>

namespace ckernel {

template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE>
ALWI void typecast_tile_init() {}

template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE>
ALWI void typecast_tile(uint32_t idst) {
    __emule_dst_check(idst, "typecast_tile");

    constexpr uint32_t DF_Float32   = 0;
    constexpr uint32_t DF_Float16_b = 5;
    constexpr uint32_t DF_Int32     = 8;
    constexpr uint32_t DF_UInt16    = 9;
    constexpr uint32_t DF_Int8      = 14;
    constexpr uint32_t DF_UInt32    = 24;
    constexpr uint32_t DF_UInt8     = 30;

    constexpr bool in_is_int = (IN_DTYPE == DF_Int32 || IN_DTYPE == DF_UInt32 ||
                                IN_DTYPE == DF_Int8 || IN_DTYPE == DF_UInt8 ||
                                IN_DTYPE == DF_UInt16);
    constexpr bool out_is_int = (OUT_DTYPE == DF_Int32 || OUT_DTYPE == DF_UInt32 ||
                                 OUT_DTYPE == DF_Int8 || OUT_DTYPE == DF_UInt8 ||
                                 OUT_DTYPE == DF_UInt16);

    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        if constexpr (in_is_int && !out_is_int) {
            // int → float: read int32 bit pattern, write as float
            int32_t iv = __emule_dst_load_i32(idst, i);
            __emule_dst[idst][i] = static_cast<float>(iv);
        } else if constexpr (!in_is_int && out_is_int) {
            // float → int: read float, store as int32 bit pattern
            float fv = __emule_dst[idst][i];
            int32_t iv = static_cast<int32_t>(std::roundf(fv));
            __emule_dst_store_i32(idst, i, iv);
        } else if constexpr (in_is_int && out_is_int) {
            // int → int: reinterpret (width narrowing/widening is a no-op in DST)
        } else {
            // float → float: bf16/f16/f32 all stored as f32 in DST — no-op
        }
    }
}

} // namespace ckernel
