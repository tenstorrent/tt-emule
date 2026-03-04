#pragma once
// Fill SFPU stubs for emulation.
// fill_tile: fill DST slot with a float value.
// fill_tile_int: fill DST slot with an integer (type-punned) value.
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void fill_tile_init() {}

ALWI void fill_tile(uint32_t idst, float param0) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = param0;
}

template <DataFormat DATA_FORMAT = DataFormat::Int32>
ALWI void fill_tile_int(uint32_t idst, uint32_t param0) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst_store_i32(idst, i, static_cast<int32_t>(param0));
}

ALWI void fill_tile_bitcast(uint32_t idst, uint32_t param0) {
    float val;
    std::memcpy(&val, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = val;
}

} // namespace ckernel
