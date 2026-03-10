#pragma once
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void div_int32_tile_init() {}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void div_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_dst_store_i32(odst, i, b != 0 ? a / b : 0);
    }
}

} // namespace ckernel
