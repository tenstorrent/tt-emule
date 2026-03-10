#pragma once
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void lcm_tile_init() {}

template<DataFormat Fmt = DataFormat::Int32>
ALWI void lcm_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        if (a == 0 || b == 0) { __emule_dst_store_i32(odst, i, 0); continue; }
        int32_t aa = a < 0 ? -a : a;
        int32_t bb = b < 0 ? -b : b;
        int32_t g = aa, h = bb;
        while (h) { int32_t t = h; h = g % h; g = t; }
        __emule_dst_store_i32(odst, i, aa / g * bb);
    }
}

} // namespace ckernel
