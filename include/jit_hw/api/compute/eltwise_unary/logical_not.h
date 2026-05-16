#pragma once
// Emulator stub for logical_not SFPU tile op.
// Result is 1.0 if input == 0, else 0.0.

#include <cstdint>
#include "jit_hw/api/compute/common_globals.h"

namespace ckernel {

ALWI void logical_not_tile_init() {}

template <DataFormat DATA_FORMAT = DataFormat::Float32>
ALWI void logical_not_tile(uint32_t idst) {
    __emule_dst_check(idst, "logical_not_tile");
    constexpr bool is_int = (DATA_FORMAT == DataFormat::Int32 ||
                             DATA_FORMAT == DataFormat::UInt32 ||
                             DATA_FORMAT == DataFormat::UInt16);
    if constexpr (is_int) {
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            int32_t v = __emule_dst_load_i32(idst, i);
            __emule_dst_store_i32(idst, i, v == 0 ? 1 : 0);
        }
    } else {
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            __emule_dst[idst][i] = (__emule_dst[idst][i] == 0.0f) ? 1.0f : 0.0f;
        }
    }
}

} // namespace ckernel
